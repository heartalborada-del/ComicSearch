/**
 * Type definitions for the pack info API.
 */

/** A keyword associated with a pack. */
export interface Keyword {
    id: number
    name: string
}

/** Pack metadata returned by GET /info/{id}. */
export interface PackInfo {
    pack_id: number
    title: string | null
    source: string | null
    keyword_ids: number[]
    keywords: Keyword[]
}
