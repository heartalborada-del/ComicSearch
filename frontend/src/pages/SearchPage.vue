<script setup lang="ts">
/**
 * Search page — image upload, search options, and results display.
 * Shows best match (highlighted) and candidate manga in a responsive grid.
 */
import { computed, ref } from 'vue'
import { useDisplay } from 'vuetify'
import { useSearchStore } from '@/stores/search'
import SearchUpload from '@/components/SearchUpload.vue'
import ConfidenceBadge from '@/components/ConfidenceBadge.vue'
import ComicCard from '@/components/ComicCard.vue'
import PagePreview from '@/components/PagePreview.vue'

const display = useDisplay()
const searchStore = useSearchStore()

const selectedFile = ref<File | null>(null)
const showOptions = ref(false)
const showPagePreview = ref(false)

/** Keyword ID input as comma-separated string. */
const keywordInput = ref('')

/** Parsed keyword IDs from input. */
const parsedKeywordIds = computed<number[]>(() => {
  const trimmed = keywordInput.value.trim()
  if (!trimmed) return []
  return trimmed
    .split(/[,，\s]+/)
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => !isNaN(n) && n > 0)
})

const canSearch = computed(() => selectedFile.value !== null && !searchStore.loading)

const hasResult = computed(() => searchStore.result !== null)
const candidates = computed(() => searchStore.result?.candidate_manga ?? [])
const bestManga = computed(() => searchStore.result?.best_manga ?? null)
const otherCandidates = computed(() => candidates.value.slice(1))

/** Responsive grid columns. */
const gridCols = computed(() => {
  if (display.xs.value) return '6'
  if (display.sm.value) return '4'
  if (display.md.value) return '3'
  return '2'
})

async function handleSearch(): Promise<void> {
  if (!selectedFile.value) return
  searchStore.params.keyword_ids = parsedKeywordIds.value
  await searchStore.executeSearch(selectedFile.value)
  showPagePreview.value = false
}

function onFileUpdate(file: File | null): void {
  selectedFile.value = file
  searchStore.clearResult()
}
</script>

<template>
  <div class="search-page">
    <!-- Upload Section -->
    <v-card variant="tonal" class="bg-surface-container mb-4 pa-4" rounded="lg">
      <SearchUpload :file="selectedFile" @update:file="onFileUpdate" />

      <!-- Options Toggle -->
      <div class="d-flex justify-center mt-3">
        <v-btn variant="text" size="small" :prepend-icon="showOptions ? 'mdi-chevron-up' : 'mdi-tune'"
          @click="showOptions = !showOptions">
          {{ showOptions ? '收起选项' : '搜索选项' }}
        </v-btn>
      </div>

      <!-- Search Options -->
      <v-expand-transition>
        <div v-show="showOptions" class="mt-3">
          <v-row dense>
            <v-col cols="12" sm="6">
              <v-switch v-model="searchStore.params.robust_partial" label="鲁棒部分匹配" density="compact" hide-details />
            </v-col>
            <v-col cols="12" sm="6">
              <v-switch v-model="searchStore.params.include_corners" label="包含角落视角" density="compact" hide-details />
            </v-col>
            <v-col cols="12" sm="6">
              <v-switch v-model="searchStore.params.include_contrast" label="包含对比度视角" density="compact" hide-details />
            </v-col>
            <v-col cols="12" sm="6">
              <v-text-field v-model="keywordInput" label="关键词 ID（逗号分隔）" density="compact" hide-details
                placeholder="例如: 1, 2, 3" />
            </v-col>
            <v-col cols="12" sm="6">
              <v-slider v-model="searchStore.params.per_view_limit" :min="10" :max="300" :step="10" label="每视角结果数"
                density="compact" />
            </v-col>
            <v-col cols="12" sm="6">
              <v-slider v-model="searchStore.params.top_k_manga" :min="1" :max="50" :step="1" label="返回漫画数"
                density="compact" />
            </v-col>
          </v-row>
        </div>
      </v-expand-transition>

      <!-- Search Button -->
      <div class="d-flex justify-center mt-4">
        <v-btn size="large" color="primary" rounded="lg" prepend-icon="mdi-magnify" :loading="searchStore.loading"
          :disabled="!canSearch" @click="handleSearch">
          搜索
        </v-btn>
      </div>
    </v-card>

    <!-- Loading State -->
    <div v-if="searchStore.loading" class="d-flex justify-center align-center py-8">
      <v-progress-circular indeterminate color="primary" size="48" width="4" />
    </div>

    <!-- Error State -->
    <v-alert v-if="searchStore.error" type="error" variant="tonal" rounded="lg" class="mb-4" closable>
      {{ searchStore.error }}
    </v-alert>

    <!-- Results -->
    <div v-if="hasResult && !searchStore.loading" class="results-section">
      <!-- Confidence + Result Summary -->
      <div class="d-flex align-center ga-3 mb-4 flex-wrap">
        <h2 class="text-h6 font-weight-medium">搜索结果</h2>
        <ConfidenceBadge :confidence="searchStore.result!.confidence" />
        <v-chip size="small" variant="tonal" color="primary">
          {{ candidates.length }} 个候选
        </v-chip>
      </div>

      <!-- No results -->
      <v-alert v-if="candidates.length === 0" type="info" variant="tonal" rounded="lg" class="mb-4">
        未找到匹配的漫画
      </v-alert>

      <!-- Pack info loading indicator -->
      <div v-if="searchStore.packInfoLoading"
        class="d-flex align-center ga-2 mb-4 text-caption text-on-surface-variant">
        <v-progress-circular indeterminate size="16" width="2" color="primary" />
        正在加载漫画详情…
      </div>

      <!-- Best Match (highlighted) -->
      <div v-if="bestManga" class="mb-6">
        <div class="text-subtitle-1 font-weight-medium mb-2">最佳匹配</div>
        <v-row>
          <v-col :cols="gridCols" sm="6" md="4">
            <ComicCard :candidate="bestManga" :pack-info="searchStore.packInfoMap[bestManga.pack_id] ?? null"
              highlighted />
          </v-col>

          <!-- Page Preview for best match -->
          <v-col v-if="bestManga.top_page_no !== null" cols="12" sm="6" md="4" lg="3">
            <div class="text-subtitle-2 mb-2">匹配页面预览</div>
            <PagePreview :pack-id="bestManga.pack_id" :page-no="bestManga.top_page_no"
              :origin-path="bestManga.top_page_origin_path" />
          </v-col>
        </v-row>
      </div>

      <!-- Other Candidates -->
      <div v-if="otherCandidates.length > 0">
        <div class="text-subtitle-1 font-weight-medium mb-2">其他候选</div>
        <v-row>
          <v-col v-for="candidate in otherCandidates" :key="candidate.pack_id" :cols="gridCols">
            <ComicCard :candidate="candidate" :pack-info="searchStore.packInfoMap[candidate.pack_id] ?? null" />
          </v-col>
        </v-row>
      </div>
    </div>

    <!-- Empty State -->
    <div v-if="!hasResult && !searchStore.loading && !searchStore.error"
      class="d-flex flex-column align-center justify-center py-12 text-center">
      <v-icon size="80" color="outline" class="mb-4">mdi-image-search</v-icon>
      <div class="text-h6 text-on-surface-variant mb-1">上传图片开始搜索</div>
      <div class="text-body-2 text-on-surface-variant">
        支持以图搜漫，找到最相似的漫画
      </div>
    </div>
  </div>
</template>
