/**
 * Pack info API module.
 * Wraps the GET /info/{id} endpoint for querying pack metadata.
 */
import { getJson } from './client'
import type { PackInfo } from '@/types/info'

/**
 * Get pack metadata by pack ID.
 * @param packId - The pack ID to query
 * @returns Pack info with title, source, and keywords
 */
export function getPackInfo(packId: number): Promise<PackInfo> {
    return getJson<PackInfo>(`/info/${packId}`)
}
