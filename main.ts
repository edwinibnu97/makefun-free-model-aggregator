// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// It reads the trial brief, probes every provider's catalog route, builds one
// catalog and registers it with the env, then serves the client's twenty
// requests: each answered by the best free model that meets its requirements,
// failing over to the next candidate when a provider refuses, and returned in
// the wire the client used, JSON or server-sent events.
//
// Routing (lib/router.ts) ranks free candidates by the quality table's rating
// and spreads load over the near-equal top providers, so no single provider's
// per-trial budget is drained. Every upstream call is a plain JSON call; the
// provider's answer is re-emitted here in the client's wire, which keeps the
// streaming event order exact. The catalog is built once and used for both
// `--list` and the register body, byte for byte.
//
// `node main.ts --list` prints that catalog and exits 0. Run outside a trial,
// with no brief and no env, it prints {"providers":[]} and exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  authHeaders,
  joinBase,
  probeProvider,
  type Catalog,
  type Probe,
} from './lib/catalog.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicResponse,
  anthropicStream,
  anthropicToAnthropic,
  anthropicToOpenAI,
  clientToGemini,
  openAIError,
  openAIModelList,
  openAIResponse,
  openAIStream,
  openAIToAnthropic,
  openAIToOpenAI,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  textOf,
  type ProviderAnswer,
  type ServedModel,
} from './lib/translate.ts';
import { Router, type Candidate } from './lib/router.ts';
import {
  isRecord,
  readBrief,
  type Json,
  type Requirement,
} from './lib/brief.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 25000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** How often the register is retried while the env is not answering. */
const REGISTER_RETRIES = 20;
const REGISTER_RETRY_MS = 500;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
let router: Router | null = null;
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;
/** The arrival order of the request now being served (for the router's cooldown). */
let requestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
  router = new Router(probes, brief.ratings);
}

const byId = (a: ServedModel, b: ServedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = brief.ratings?.ratings.find((r) => r.provider === probe.provider.id && r.model === model.id) ?? null;
      const free = row !== null ? row.free : (model.free ?? false);
      if (!free) continue;
      out.push({ id: model.id, provider: probe.provider.id, displayName: model.displayName });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** Whether this client request asks for JSON mode on the OpenAI wire. */
function jsonModeOf(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  if (!isRecord(body.response_format)) return false;
  const format = body.response_format.type;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** The error message a provider's refusal body carries, in any wire's shape. */
function errorMessageOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value)) {
      if (isRecord(value.error) && typeof value.error.message === 'string') return value.error.message;
      if (typeof value.message === 'string') return value.message;
    }
  } catch {
    // Not JSON: the raw text is the message.
  }
  return text.slice(0, 300);
}

interface CallOutcome {
  ok: true;
  answer: ProviderAnswer;
}
interface CallFailure {
  ok: false;
  message: string;
}
type CallResult = CallOutcome | CallFailure;

/**
 * One JSON call to a candidate provider, in its own wire. Returns the
 * provider's answer, or the refusal (a 402, 429, 5xx, a 400 or a network
 * error) with its message, so the router can decide what it means.
 */
async function callProvider(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<CallResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const model = candidate.model.id;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? openAIToOpenAI(body, model) : anthropicToOpenAI(body, model);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'anthropic' ? anthropicToAnthropic(body, model) : openAIToAnthropic(body, model);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    upstreamBody = clientToGemini(clientApi, body);
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, message: (err as Error).message };
  }
  const text = await res.text();
  if (!res.ok) return { ok: false, message: errorMessageOf(text) };
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, message: 'the provider answered a body that is not JSON' };
  }
  const answer =
    provider.wire === 'openai'
      ? parseOpenAIAnswer(parsed)
      : provider.wire === 'anthropic'
        ? parseAnthropicAnswer(parsed)
        : parseGeminiAnswer(parsed);
  if (answer === null) return { ok: false, message: 'the provider answered an unparseable body' };
  return { ok: true, answer };
}

/** Serve one client request: try its candidates in order until one answers. */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<{ candidate: Candidate; answer: ProviderAnswer } | null> {
  if (router === null) return null;
  const index = requestIndex;
  requestIndex += 1;
  const requires = requirementsFor(body);
  const jsonMode = jsonModeOf(clientApi, body);
  const candidates = router.candidatesFor(requires, jsonMode, clientApi, index);
  for (const candidate of candidates) {
    router.noteCall(candidate.probe.provider.id);
    const result = await callProvider(candidate, clientApi, body);
    if (result.ok) return { candidate, answer: result.answer };
    router.noteRefusal(candidate.probe.provider.id, result.message, index);
    log(`${candidate.probe.provider.id}/${candidate.model.id} refused: ${result.message.slice(0, 120)}`);
  }
  return null;
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const served = await serve('openai', body);
  if (served === null) {
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  }
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
    res.end(openAIStream(served.candidate.model.id, served.answer));
    return;
  }
  sendJson(res, 200, openAIResponse(served.candidate.model.id, served.answer));
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const served = await serve('anthropic', body);
  if (served === null) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  }
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
    res.end(anthropicStream(served.candidate.model.id, served.answer));
    return;
  }
  sendJson(res, 200, anthropicResponse(served.candidate.model.id, served.answer));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < REGISTER_RETRIES; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) {
        throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      }
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the workload holds ${reply.requests} requests`);
      }
      return;
    } catch (err) {
      lastError = err as Error;
      await sleep(REGISTER_RETRY_MS);
    }
  }
  throw lastError ?? new Error('register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
