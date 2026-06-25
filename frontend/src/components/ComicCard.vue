<script setup lang="ts">
/**
 * Comic card — displays a manga candidate with cover thumbnail,
 * score progress bar, match statistics, and pack metadata (title, keywords)
 * fetched via the linked pack-info query.
 * Clicking the card navigates to the detail page.
 */
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { coverUrl } from '@/utils/image-url'
import type { MangaCandidate } from '@/types/search'
import type { PackInfo } from '@/types/info'

const props = withDefaults(
  defineProps<{
    candidate: MangaCandidate
    /** Pack info from the linked query (search store fetches this after search). */
    packInfo?: PackInfo | null
    highlighted?: boolean
  }>(),
  {
    packInfo: null,
    highlighted: false,
  },
)

const router = useRouter()

const scorePercent = computed(() => Math.round(props.candidate.score * 100))
const top1Percent = computed(() => Math.round(props.candidate.top1_score * 100))

const coverSrc = computed(() => coverUrl(props.candidate.pack_id))

const displayTitle = computed(() => props.packInfo?.title || `Pack #${props.candidate.pack_id}`)
const hasTitle = computed(() => !!props.packInfo?.title)
const keywords = computed(() => props.packInfo?.keywords ?? [])

function goToDetail(): void {
  router.push({ name: 'detail', params: { id: props.candidate.pack_id } })
}
</script>

<template>
  <v-card class="comic-card state-layer cursor-pointer h-100"
    :class="highlighted ? 'bg-primary-container' : 'bg-surface-container'" variant="tonal" @click="goToDetail">
    <!-- Cover Image -->
    <v-img :src="coverSrc" aspect-ratio="0.72" cover class="bg-surface-container-high">
      <template #placeholder>
        <div class="d-flex align-center justify-center fill-height">
          <v-icon size="36" color="outline">mdi-book-outline</v-icon>
        </div>
      </template>
      <template #error>
        <div class="d-flex align-center justify-center fill-height">
          <v-icon size="36" color="outline">mdi-book-off-outline</v-icon>
        </div>
      </template>

      <!-- Highlighted badge -->
      <v-chip v-if="highlighted" color="primary" size="x-small" class="ma-1" prepend-icon="mdi-star">
        最佳匹配
      </v-chip>
    </v-img>

    <v-card-text class="pt-2 px-2 pb-2">
      <!-- Title (from linked pack info) -->
      <div class="text-body-2 font-weight-medium mb-0 text-truncate" :title="displayTitle">
        {{ displayTitle }}
      </div>

      <!-- Pack ID (secondary, shown when title exists) -->
      <div v-if="hasTitle" class="text-caption text-on-surface-variant mb-1">
        #{{ candidate.pack_id }}
      </div>

      <!-- Keywords (from linked pack info) -->
      <div v-if="keywords.length > 0" class="d-flex flex-wrap ga-1 mb-1">
        <v-chip v-for="kw in keywords.slice(0, 3)" :key="kw.id" size="x-small" variant="tonal" color="secondary">
          {{ kw.name }}
        </v-chip>
        <v-chip v-if="keywords.length > 3" size="x-small" variant="text" color="on-surface-variant">
          +{{ keywords.length - 3 }}
        </v-chip>
      </div>

      <!-- Score progress bar -->
      <div class="mb-1">
        <div class="d-flex justify-space-between text-caption text-on-surface-variant mb-0">
          <span>综合评分</span>
          <span class="font-weight-medium">{{ scorePercent }}%</span>
        </div>
        <v-progress-linear :model-value="scorePercent" color="primary" height="4" rounded />
      </div>

      <!-- Statistics -->
      <div class="d-flex ga-1 flex-wrap">
        <v-chip size="x-small" variant="tonal" color="primary">
          <v-icon start size="10">mdi-target</v-icon>
          Top1: {{ top1Percent }}%
        </v-chip>
        <v-chip size="x-small" variant="tonal" color="secondary">
          <v-icon start size="10">mdi-counter</v-icon>
          {{ candidate.hits }}
        </v-chip>
        <v-chip v-if="candidate.top_page_no !== null" size="x-small" variant="tonal" color="tertiary">
          <v-icon start size="10">mdi-file-image</v-icon>
          p{{ candidate.top_page_no }}
        </v-chip>
      </div>
    </v-card-text>
  </v-card>
</template>

<style scoped lang="scss">
.comic-card {
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.comic-card:hover {
  transform: translateY(-2px);
}

.cursor-pointer {
  cursor: pointer;
}
</style>
