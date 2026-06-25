<script setup lang="ts">
/**
 * Tag search page — fuzzy search keywords, view pack count, copy keyword ID.
 * Uses virtual scrolling for large result sets.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { getJson, ApiError } from '@/api/client'

// ---- Types ----
interface TagItem {
    id: number
    name: string
    pack_count: number
}

// ---- State ----
const searchQuery = ref('')
const tags = ref<TagItem[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const searched = ref(false)
const copiedId = ref<number | null>(null)
let _debounceTimer: ReturnType<typeof setTimeout> | null = null

const hasResults = computed(() => tags.value.length > 0)

// ---- Virtual scroll ----
const ITEM_HEIGHT = 44
const OVERSCAN = 6
const listContainer = ref<HTMLElement | null>(null)
const scrollTop = ref(0)
const viewHeight = ref(600)

const visibleCount = computed(() => Math.ceil(viewHeight.value / ITEM_HEIGHT) + OVERSCAN)
const startIdx = computed(() => Math.max(0, Math.floor(scrollTop.value / ITEM_HEIGHT) - OVERSCAN))
const endIdx = computed(() => Math.min(tags.value.length, startIdx.value + visibleCount.value + OVERSCAN))
const visibleTags = computed(() => tags.value.slice(startIdx.value, endIdx.value))
const totalHeight = computed(() => `${tags.value.length * ITEM_HEIGHT}px`)
const offsetY = computed(() => `${startIdx.value * ITEM_HEIGHT}px`)

function onScroll(): void {
    if (listContainer.value) {
        scrollTop.value = listContainer.value.scrollTop
    }
}

watch(hasResults, (v) => {
    if (v) {
        scrollTop.value = 0
        if (listContainer.value) listContainer.value.scrollTop = 0
    }
})

onMounted(() => {
    if (listContainer.value) {
        viewHeight.value = listContainer.value.clientHeight
        const ro = new ResizeObserver(([entry]) => {
            if (entry) viewHeight.value = entry.contentRect.height
        })
        ro.observe(listContainer.value)
    }
})

// ---- Actions ----
function doSearch(): void {
    if (_debounceTimer) clearTimeout(_debounceTimer)
    _debounceTimer = setTimeout(_executeSearch, 250)
}

async function _executeSearch(): Promise<void> {
    const q = searchQuery.value.trim()
    if (!q) {
        tags.value = []
        searched.value = false
        return
    }
    loading.value = true
    error.value = null
    searched.value = true
    try {
        const params = new URLSearchParams({ q, limit: '200' })
        tags.value = await getJson<TagItem[]>(`/tag/search?${params}`)
    } catch (err) {
        error.value = err instanceof ApiError ? err.detail : '搜索失败'
        tags.value = []
    } finally {
        loading.value = false
    }
}

function copyId(id: number): void {
    navigator.clipboard.writeText(String(id)).then(() => {
        copiedId.value = id
        setTimeout(() => { copiedId.value = null }, 1500)
    })
}
</script>

<template>
    <div class="tag-search-page" style="min-height: 400px">
        <!-- Header -->
        <div class="d-flex align-center ga-3 mb-4 flex-wrap">
            <h1 class="text-h5 font-weight-medium">标签搜索</h1>
            <v-chip v-if="hasResults" size="small" variant="tonal" color="primary">
                {{ tags.length }} 个结果
            </v-chip>
        </div>

        <!-- Search Input -->
        <v-card variant="tonal" class="bg-surface-container mb-4" rounded="lg">
            <v-card-text class="pa-3">
                <v-text-field v-model="searchQuery" label="搜索标签名（模糊匹配）" density="compact" variant="outlined"
                    hide-details clearable prepend-inner-icon="mdi-magnify" @input="doSearch"
                    @keydown.enter="doSearch" />
                <div class="text-caption text-on-surface-variant mt-2">
                    输入标签名搜索对应 Keyword ID，可直接用于搜索页关键词筛选
                </div>
            </v-card-text>
        </v-card>

        <!-- Loading -->
        <div v-if="loading" class="d-flex justify-center py-12">
            <v-progress-circular indeterminate color="primary" size="40" width="4" />
        </div>

        <!-- Error -->
        <v-alert v-else-if="error" type="error" variant="tonal" rounded="lg" class="mb-4">
            {{ error }}
        </v-alert>

        <!-- Empty (after search) -->
        <v-card v-else-if="searched && !hasResults" variant="tonal" class="bg-surface-container" rounded="lg">
            <v-card-text class="text-center py-8 text-on-surface-variant">
                未找到匹配的标签
            </v-card-text>
        </v-card>

        <!-- Results (virtual scroll) -->
        <v-card v-else-if="hasResults" variant="tonal" class="bg-surface-container" rounded="lg">
            <!-- Header -->
            <div class="d-flex align-center text-caption text-on-surface-variant px-3 py-2"
                style="min-height: 32px; border-bottom: 1px solid rgba(var(--v-border-color), var(--v-border-opacity))">
                <span style="width: 90px; flex-shrink: 0">ID</span>
                <span style="flex: 1">标签名</span>
                <span style="width: 80px; text-align: center">Pack 数</span>
                <span style="width: 80px; text-align: right">操作</span>
            </div>

            <!-- Virtual scroll container -->
            <div ref="listContainer" class="scroll-container"
                style="max-height: 70vh; overflow-y: auto; position: relative" @scroll="onScroll">
                <div :style="{ height: totalHeight }">
                    <div :style="{ transform: `translateY(${offsetY})` }">
                        <div v-for="tag in visibleTags" :key="tag.id" class="d-flex align-center px-3"
                            style="height: 44px">
                            <code class="text-caption text-primary" style="width: 90px; flex-shrink: 0">
                                {{ tag.id }}
                            </code>
                            <span class="text-body-2 text-truncate" style="flex: 1" :title="tag.name">
                                {{ tag.name }}
                            </span>
                            <span style="width: 80px; text-align: center">
                                <v-chip v-if="tag.pack_count > 0" size="x-small" variant="tonal" color="secondary">
                                    {{ tag.pack_count }}
                                </v-chip>
                                <span v-else class="text-caption text-on-surface-variant">0</span>
                            </span>
                            <span style="width: 80px; text-align: right">
                                <v-btn size="x-small" variant="tonal" color="primary" @click="copyId(tag.id)">
                                    <v-icon start size="14">
                                        {{ copiedId === tag.id ? 'mdi-check' : 'mdi-content-copy' }}
                                    </v-icon>
                                    {{ copiedId === tag.id ? '已复制' : '复制' }}
                                </v-btn>
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </v-card>

        <!-- Initial State -->
        <div v-if="!searched && !loading && !error"
            class="d-flex flex-column align-center justify-center py-12 text-center">
            <v-icon size="64" color="outline" class="mb-4">mdi-tag-search</v-icon>
            <div class="text-body-1 text-on-surface-variant">
                搜索标签名获取对应 Keyword ID
            </div>
        </div>
    </div>
</template>
