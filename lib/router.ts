// Routing: which free model serves one request, and what happens when the
// provider it picked refuses.
//
// Candidates are every free model that meets the request's requirements,
// ranked by the quality table's rating, best first. The ranking is pure
// rating order: on the three practice trials that measured the fewest
// provider calls at the same score (the tie-break), because the fallback
// providers stay fresh for failover while the best provider is still
// answering. A provider that refuses is read by its error message (below),
// and a provider whose budget is spent or whose outage began is never tried
// again, so the late requests of a trial fail over at once instead of
// queuing on a dead provider.
//   - "free-tier budget is spent" or "is out" (outage): the provider is dead
//     for the rest of the trial and is never tried again;
//   - anything else (a scheduled refusal, a 429, a 5xx, a network error):
//     transient. The request fails over to the next candidate at once, and
//     the provider sits out the very next request (scheduled refusals often
//     come in pairs), then is usable again.
//
// A JSON-mode request from an OpenAI client (response_format) can only go to
// a wire that carries it: OpenAI (response_format) or Gemini
// (responseMimeType). An anthropic-wire provider echoes json_mode false, so
// it is never a candidate for one. An Anthropic client expresses JSON mode
// only in its text, so any wire serves those requests.
//
// Every choice here is a function of the brief, the requests so far and the
// providers' answers: no clock, no unseeded randomness, so a replay makes
// the same choices.

import { capabilitiesOf, modelFree, type Probe, type ProbedModel } from './catalog.ts';
import { ratingOfModel, type RatingsTable, type Requirement, type Wire } from './brief.ts';

/** Milli-points of effective rating one provider call costs its models. */
const LOAD_PENALTY_MILLI = 0;

/** One (provider, model) pair that may serve a request. */
export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  ratingMilli: number;
}

/** Per-provider trial state: what it has been asked, and what it answered. */
export interface ProviderState {
  calls: number;
  dead: boolean;
  failedAtRequest: number;
}

export class Router {
  private readonly probes: Probe[];
  private readonly ratings: RatingsTable | null;
  private readonly state = new Map<string, ProviderState>();

  constructor(probes: Probe[], ratings: RatingsTable | null) {
    this.probes = probes;
    this.ratings = ratings;
    for (const probe of probes) {
      this.state.set(probe.provider.id, { calls: 0, dead: false, failedAtRequest: -1 });
    }
  }

  private stateOf(providerId: string): ProviderState {
    let state = this.state.get(providerId);
    if (state === undefined) {
      state = { calls: 0, dead: false, failedAtRequest: -1 };
      this.state.set(providerId, state);
    }
    return state;
  }

  /** Whether a model's capabilities meet one request's requirements. */
  private static meets(requires: Requirement, model: ProbedModel, row: ReturnType<typeof ratingOfModel>): boolean {
    const caps = capabilitiesOf(model, row);
    if (requires.tools === true && caps.tools !== true) return false;
    if (requires.vision === true && caps.vision !== true) return false;
    if (requires.json === true && caps.json !== true) return false;
    if (requires.minContext !== undefined && (caps.contextLength === null || caps.contextLength < requires.minContext)) {
      return false;
    }
    return true;
  }

  /**
   * The candidates for one request, best first. `jsonMode` is true when the
   * client asked for JSON mode on the OpenAI wire; `requestIndex` is the
   * arrival order, used for the one-request cooldown after a refusal.
   */
  candidatesFor(
    requires: Requirement,
    jsonMode: boolean,
    clientWire: 'openai' | 'anthropic',
    requestIndex: number,
  ): Candidate[] {
    const out: Candidate[] = [];
    for (const probe of this.probes) {
      if (!probe.ok) continue;
      // A JSON-mode OpenAI request cannot go to an anthropic-wire provider:
      // its wire cannot carry response_format, so the echo would say absent.
      if (jsonMode && clientWire === 'openai' && probe.provider.wire === 'anthropic') continue;
      for (const model of probe.models) {
        const row = ratingOfModel(this.ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        if (!Router.meets(requires, model, row)) continue;
        out.push({ probe, model, ratingMilli: row?.ratingMilli ?? 0 });
      }
    }
    const byEffective = (c: Candidate): number =>
      c.ratingMilli - LOAD_PENALTY_MILLI * this.stateOf(c.probe.provider.id).calls;
    out.sort((a, b) => {
      const eff = byEffective(b) - byEffective(a);
      if (eff !== 0) return eff;
      if (b.ratingMilli !== a.ratingMilli) return b.ratingMilli - a.ratingMilli;
      if (a.probe.provider.id !== b.probe.provider.id) {
        return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
      }
      return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
    });
    // A provider that refused on the previous request sits this one out,
    // unless it is the only way to serve it.
    const cooled = out.filter((c) => this.stateOf(c.probe.provider.id).failedAtRequest !== requestIndex - 1);
    const usable = cooled.length > 0 ? cooled : out;
    // Dead providers are never tried, unless nothing else is left.
    const alive = usable.filter((c) => !this.stateOf(c.probe.provider.id).dead);
    return alive.length > 0 ? alive : usable;
  }

  /** Record that a call to `providerId` was made (it counts, answered or refused). */
  noteCall(providerId: string): void {
    this.stateOf(providerId).calls += 1;
  }

  /**
   * Record a refusal. Returns true when the provider is dead for the trial
   * (its budget is spent or its outage began); false for a transient refusal,
   * which also starts the one-request cooldown.
   */
  noteRefusal(providerId: string, message: string, requestIndex: number): boolean {
    const state = this.stateOf(providerId);
    if (message.includes('budget is spent') || message.includes(' is out')) {
      state.dead = true;
      return true;
    }
    state.failedAtRequest = requestIndex;
    return false;
  }

  /** The wire of one candidate's provider. */
  static wireOf(candidate: Candidate): Wire {
    return candidate.probe.provider.wire;
  }
}
