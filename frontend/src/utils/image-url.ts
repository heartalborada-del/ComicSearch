/**
 * Image URL builder utility.
 * Constructs full image URLs from VITE_IMAGE_BASE_URL for:
 * - Cover thumbnails (by pack_id) → /served/cover/{pack_id}
 * - Matched page previews (from Qdrant origin_source_path) → /{originPath}
 * - Legacy: page previews (by pack_id + page_no)
 */

const IMAGE_BASE_URL = import.meta.env.VITE_IMAGE_BASE_URL || ''

/**
 * Build a cover thumbnail URL for a given pack.
 * @param packId - The pack ID
 * @returns Full URL to the cover thumbnail image
 */
export function coverUrl(packId: number): string {
    return `${IMAGE_BASE_URL}/served/cover/${packId}`
}

/**
 * Build an origin image URL from a relative path stored in Qdrant.
 * @param originPath - Relative path from Caddy root (e.g. "origin/ehentai/389-f805/page_0001.jpg")
 * @returns Full URL to the origin page image
 */
export function originImageUrl(originPath: string | null | undefined): string {
    if (!originPath) return ''
    if (originPath.startsWith('http://') || originPath.startsWith('https://')) return originPath
    return `${IMAGE_BASE_URL}/${originPath}`
}

/**
 * Build a page preview URL for a specific page in a pack (legacy fallback).
 * @param packId - The pack ID
 * @param pageNo - The page number (1-indexed)
 * @returns Full URL to the page image
 */
export function pageUrl(packId: number, pageNo: number): string {
    return `${IMAGE_BASE_URL}/served/page/${packId}/${pageNo}`
}

/**
 * Build a page thumbnail URL for a specific page in a pack.
 * @param packId - The pack ID
 * @param pageNo - The page number (1-indexed)
 * @returns Full URL to the page thumbnail image
 */
export function pageThumbUrl(packId: number, pageNo: number): string {
    return `${IMAGE_BASE_URL}/served/page/${packId}/${pageNo}/thumb`
}

/**
 * Build an image URL from a relative path (e.g. cover_thumb_path from search results).
 * @param path - Relative path from search payload
 * @returns Full URL to the image
 */
export function pathUrl(path: string): string {
    if (!path) return ''
    if (path.startsWith('http://') || path.startsWith('https://')) return path
    const separator = IMAGE_BASE_URL.endsWith('/') ? '' : '/'
    return `${IMAGE_BASE_URL}${separator}${path}`
}

/**
 * Get the raw image base URL for external use.
 */
export function getImageBaseUrl(): string {
    return IMAGE_BASE_URL
}
