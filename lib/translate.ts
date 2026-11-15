// The client wires and the three upstream wires.
//
// The trial's client calls this server as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a JSON
// body or with server-sent events. This server always calls its providers
// with a plain JSON body (never a stream): the provider's JSON answer carries
// the same text and usage its stream would, and this server re-emits the
// answer in the wire the client asked for, streamed or not. That keeps every
// event sequence under this file's control.
//
// The provider echoes what it parsed, and the client compares that echo with
// what it sent, so every translation below passes the request through
// untouched: the system prompt, the user text (images included), the tools,
// the stop sequences, max_tokens and JSON mode. A field rewritten or dropped
// loses the request.
//
// JSON mode: an OpenAI request carries response_format; a Gemini request
// carries generationConfig.responseMimeType. An Anthropic request cannot carry
// it at all (its provider echoes json_mode false), so a JSON-mode OpenAI
// request is never routed to an anthropic-wire provider (the router enforces
// that; see lib/router.ts).
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so the same body and the same provider answer give the same
// bytes on a replay. The model listings use fixed timestamps for the same
// reason: a `created` that moved between runs would be a nondeterministic
// document.

import { isRecord, type Json } from './brief.ts';

/** One model this server serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.displayName,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
    })),
    has_more: false,
  };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The max_tokens a body carries, on either wire's spelling. */
function maxTokensOf(body: Json): number | undefined {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return undefined;
}

/** The stop sequences a body carries, on either wire's spelling. */
function stopsOf(body: Json): string[] | undefined {
  if (Array.isArray(body.stop)) {
    const stops = body.stop.filter((s): s is string => typeof s === 'string');
    return stops.length > 0 ? stops : undefined;
  }
  if (Array.isArray(body.stop_sequences)) {
    const stops = body.stop_sequences.filter((s): s is string => typeof s === 'string');
    return stops.length > 0 ? stops : undefined;
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Client request -> OpenAI provider request
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part: text or an inline image. */
function openAIPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, and tools, tool_choice, the output cap, temperature,
 * top_p and stop sequences pass through. JSON mode has no Anthropic spelling,
 * so none is sent.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = openAIPartOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages, stream: false };
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOf(body);
  if (stops !== undefined) out.stop = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  return out;
}

/** An OpenAI Chat Completions request as an OpenAI provider request naming `model`. */
export function openAIToOpenAI(body: Json, model: string): Json {
  const out: Json = { ...body, model, stream: false };
  delete out.stream_options;
  return out;
}

/** An Anthropic Messages request as an Anthropic provider request naming `model`. */
export function anthropicToAnthropic(body: Json, model: string): Json {
  const out: Json = { ...body, model, stream: false };
  return out;
}

// ---------------------------------------------------------------------------
// Client request -> Anthropic provider request
// ---------------------------------------------------------------------------

/** One OpenAI content part as an Anthropic content block: text or an image. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return {
        type: 'image',
        source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] },
      };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: system messages become the system prompt, image_url parts become
 * image blocks, tool_calls become tool_use blocks, tool messages become
 * tool_result blocks, and tools, tool_choice, the output cap, temperature,
 * top_p and stop sequences pass through. response_format has no Anthropic
 * spelling and is dropped: never route a JSON-mode OpenAI request here.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role !== 'user') continue;
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      const block = anthropicBlockOf(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages, stream: false };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOf(body);
  if (stops !== undefined) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters)
            ? raw.function.parameters
            : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
    else if (choice === 'none') out.tool_choice = { type: 'none' };
  } else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
    if (typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Client request -> Gemini provider request
// ---------------------------------------------------------------------------

/** The media type and base64 data of a data URL, or null when it is not one. */
function dataUrlParts(url: string): { media: string; data: string } | null {
  const match = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { media: match[1] !== '' ? match[1] : 'image/png', data: match[2] };
}

/** One OpenAI content part as Gemini parts. */
function geminiPartsOfOpenAI(part: Json): Json[] {
  if (part.type === 'text' && typeof part.text === 'string') return [{ text: part.text }];
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const data = dataUrlParts(part.image_url.url);
    if (data !== null) return [{ inlineData: { mimeType: data.media, data: data.data } }];
    return [{ fileData: { mimeType: 'image/png', fileUri: part.image_url.url } }];
  }
  return [];
}

/** One Anthropic content block as Gemini parts. */
function geminiPartsOfAnthropic(block: Json): Json[] {
  if (block.type === 'text' && typeof block.text === 'string') return [{ text: block.text }];
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return [
        {
          inlineData: {
            mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png',
            data: source.data,
          },
        },
      ];
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return [{ fileData: { mimeType: 'image/png', fileUri: source.url } }];
    }
  }
  return [];
}

/** Tool definitions as Gemini function declarations, from either wire's spelling. */
function geminiDeclarationsOf(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  const out: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw)) continue;
    if (typeof raw.name === 'string') {
      // Anthropic spelling.
      out.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
      continue;
    }
    if (isRecord(raw.function) && typeof raw.function.name === 'string') {
      // OpenAI spelling.
      out.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters)
          ? raw.function.parameters
          : { type: 'object', properties: {} },
      });
    }
  }
  return out;
}

/**
 * A client request (either wire) as a Gemini generateContent body for `model`.
 * The system prompt becomes systemInstruction, user turns become user contents
 * (text plus inlineData for images), assistant turns become model contents
 * (text plus functionCall for tool calls), tool results become
 * functionResponse parts, and tools, the output cap, stop sequences,
 * temperature and JSON mode (responseMimeType) land in generationConfig.
 * The model itself travels in the request path, not the body.
 */
export function clientToGemini(clientApi: 'openai' | 'anthropic', body: Json): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];

  if (clientApi === 'anthropic') {
    const system = textOf(body.system);
    if (system !== '') systemParts.push(system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'user') {
        if (typeof raw.content === 'string') {
          contents.push({ role: 'user', parts: [{ text: raw.content }] });
          continue;
        }
        if (!Array.isArray(raw.content)) continue;
        const parts: Json[] = [];
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'tool_result') {
            parts.push({
              functionResponse: {
                name: '',
                response: { content: toolResultText(block.content) },
              },
            });
            continue;
          }
          parts.push(...geminiPartsOfAnthropic(block));
        }
        if (parts.length > 0) contents.push({ role: 'user', parts });
        continue;
      }
      if (raw.role !== 'assistant' || !Array.isArray(raw.content)) continue;
      const parts: Json[] = [];
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'tool_use') {
          parts.push({
            functionCall: {
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            },
          });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
    }
  } else {
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') systemParts.push(text);
        continue;
      }
      if (raw.role === 'tool') {
        contents.push({
          role: 'user',
          parts: [
            {
              functionResponse: {
                name: '',
                response: { content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) },
              },
            },
          ],
        });
        continue;
      }
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        const text = textOf(raw.content);
        if (text !== '') parts.push({ text });
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({
            functionCall: {
              name: typeof call.function.name === 'string' ? call.function.name : '',
              args: argumentsOf(call.function.arguments),
            },
          });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (raw.role !== 'user') continue;
      if (typeof raw.content === 'string') {
        contents.push({ role: 'user', parts: [{ text: raw.content }] });
        continue;
      }
      if (!Array.isArray(raw.content)) continue;
      const parts: Json[] = [];
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        parts.push(...geminiPartsOfOpenAI(part));
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const declarations = geminiDeclarationsOf(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  const stops = stopsOf(body);
  if (stops !== undefined) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (clientApi === 'openai' && isRecord(body.response_format)) {
    const format = body.response_format.type;
    if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Provider JSON answers -> { text, usage }
// ---------------------------------------------------------------------------

export interface ProviderAnswer {
  text: string;
  usage: { input: number; output: number };
}

/** A Chat Completions JSON answer. */
export function parseOpenAIAnswer(body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) },
  };
}

/** A Messages JSON answer. */
export function parseAnthropicAnswer(body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  if (!Array.isArray(body.content)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textOf(body.content),
    usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) },
  };
}

/** A Gemini generateContent JSON answer. */
export function parseGeminiAnswer(body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    usage: { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) },
  };
}

// ---------------------------------------------------------------------------
// Provider answers -> client responses (JSON and SSE)
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into the 13-character pieces a streamed answer carries. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** A provider answer as an OpenAI Chat Completions JSON object. */
export function openAIResponse(model: string, answer: ProviderAnswer): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: 'stop',
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    },
  };
}

/** A provider answer as an OpenAI Chat Completions SSE stream. */
export function openAIStream(model: string, answer: ProviderAnswer): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent(null, chunk({ content: piece }, null)));
  }
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.usage.input,
        completion_tokens: answer.usage.output,
        total_tokens: answer.usage.input + answer.usage.output,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A provider answer as an Anthropic Messages JSON object. */
export function anthropicResponse(model: string, answer: ProviderAnswer): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

/** A provider answer as an Anthropic Messages SSE stream. */
export function anthropicStream(model: string, answer: ProviderAnswer): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: answer.usage.output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
