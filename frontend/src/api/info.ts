/**
 * Pack info API module.
 * Wraps the GET /info/{id} and GET /stats endpoints.
 */
import { getJson } from './client'
import type { PackInfo, StatsResponse } from '@/types/info'

/**
 * Get pack metadata by pack ID.
 * @param packId - The pack ID to query
 * @returns Pack info with title, source, and keywords
 */
export function getPackInfo(packId: number): Promise<PackInfo> {
    return getJson<PackInfo>(`/info/${packId}`)
}

/**
 * Get global statistics (pack count and keyword count).
 */
export function getStats(): Promise<StatsResponse> {
    return getJson<StatsResponse>('/stats')
}
