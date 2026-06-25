/**
 * Type definitions for the search API.
 */

/** Confidence level returned by the search endpoint. */
export type ConfidenceLevel = 'high' | 'medium' | 'low'

/** A manga candidate in search results. */
export interface MangaCandidate {
    pack_id: number
    score: number
    hits: number
    top1_score: number
    top_page_no: number | null
    /** Relative path to the origin page image (e.g. "origin/ehentai/389-f805/page_0001.jpg"). */
    top_page_origin_path: string | null
}

/** Full search response from POST /search. */
export interface SearchResponse {
    best_manga: MangaCandidate | null
    confidence: ConfidenceLevel
    candidate_manga: MangaCandidate[]
}

/** Parameters for a search request. */
export interface SearchParams {
    robust_partial?: boolean
    include_corners?: boolean
    include_contrast?: boolean
    per_view_limit?: number
    top_k_manga?: number
    keyword_ids?: number[]
}
