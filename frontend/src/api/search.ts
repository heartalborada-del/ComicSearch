/**
 * Search API module.
 * Wraps the POST /search endpoint for image-based manga search.
 */
import { postForm } from './client'
import type { SearchParams, SearchResponse } from '@/types/search'

/**
 * Search for manga by uploading an image.
 * @param image - The image file to search with
 * @param params - Optional search parameters
 * @returns Search response with best match and candidates
 */
export function searchImage(image: File, params?: SearchParams): Promise<SearchResponse> {
    const formData = new FormData()
    formData.append('image', image)

    if (params?.robust_partial !== undefined) {
        formData.append('robust_partial', String(params.robust_partial))
    }
    if (params?.include_corners !== undefined) {
        formData.append('include_corners', String(params.include_corners))
    }
    if (params?.include_contrast !== undefined) {
        formData.append('include_contrast', String(params.include_contrast))
    }
    if (params?.per_view_limit !== undefined) {
        formData.append('per_view_limit', String(params.per_view_limit))
    }
    if (params?.top_k_manga !== undefined) {
        formData.append('top_k_manga', String(params.top_k_manga))
    }
    if (params?.keyword_ids && params.keyword_ids.length > 0) {
        formData.append('keyword_ids', JSON.stringify(params.keyword_ids))
    }

    return postForm<SearchResponse>('/search', formData, { timeoutMs: 60_000 })
}
