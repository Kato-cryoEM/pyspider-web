/**
 * Tutorial loader.
 *
 * Lessons live as JSON files on the backend (`backend/tutorials/*.json`).
 * The frontend fetches them at startup via `/api/tutorials`. This lets
 * anyone author a lesson without touching TypeScript / rebuilding.
 *
 * See `docs/tutorial_authoring.md` for the lesson-file schema.
 */

import { fetchTutorials } from "../api";
import type { Lesson } from "./types";

let cache: Lesson[] | null = null;
let inflight: Promise<Lesson[]> | null = null;

/** Fetch (and memoise) the tutorial catalogue. */
export async function loadLessons(): Promise<Lesson[]> {
  if (cache) return cache;
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const { lessons, errors } = await fetchTutorials();
      if (errors && (errors as unknown[]).length) {
        console.warn("tutorial load errors:", errors);
      }
      cache = (lessons as Lesson[]) ?? [];
    } catch (e) {
      console.error("tutorial fetch failed:", e);
      cache = [];
    } finally {
      inflight = null;
    }
    return cache!;
  })();
  return inflight;
}

/** Synchronous access — returns [] until `loadLessons()` has resolved. */
export function getLessons(): Lesson[] {
  return cache ?? [];
}
