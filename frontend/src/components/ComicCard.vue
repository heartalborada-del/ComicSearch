<script setup lang="ts">
/**
 * Comic card — displays a manga candidate with cover thumbnail,
 * score progress bar, and match statistics.
 * Clicking the card navigates to the detail page.
 */
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { coverUrl } from '@/utils/image-url'
import type { MangaCandidate } from '@/types/search'

const props = withDefaults(
  defineProps<{
    candidate: MangaCandidate
    highlighted?: boolean
  }>(),
  {
    highlighted: false,
  },
)

const router = useRouter()

const scorePercent = computed(() => Math.round(props.candidate.score * 100))
const top1Percent = computed(() => Math.round(props.candidate.top1_score * 100))

const coverSrc = computed(() => coverUrl(props.candidate.pack_id))

function goToDetail(): void {
  router.push({ name: 'detail', params: { id: props.candidate.pack_id } })
}
</script>

<template>
  <v-card
    class="comic-card state-layer cursor-pointer h-100"
    :class="highlighted ? 'bg-primary-container' : 'bg-surface-container'"
    variant="tonal"
    @click="goToDetail"
  >
    <!-- Cover Image -->
    <v-img
      :src="coverSrc"
      aspect-ratio="0.7"
      cover
      class="bg-surface-container-high"
    >
      <template #placeholder>
        <div class="d-flex align-center justify-center fill-height">
          <v-icon size="48" color="outline">mdi-book-outline</v-icon>
        </div>
      </template>
      <template #error>
        <div class="d-flex align-center justify-center fill-height">
          <v-icon size="48" color="outline">mdi-book-off-outline</v-icon>
        </div>
      </template>

      <!-- Highlighted badge -->
      <v-chip
        v-if="highlighted"
        color="primary"
        size="x-small"
        class="ma-2"
        prepend-icon="mdi-star"
      >
        最佳匹配
      </v-chip>
    </v-img>

    <v-card-text class="pt-3">
      <!-- Pack ID -->
      <div class="text-body-2 text-on-surface-variant mb-2">
        Pack #{{ candidate.pack_id }}
      </div>

      <!-- Score progress bar -->
      <div class="mb-2">
        <div class="d-flex justify-space-between text-caption text-on-surface-variant mb-1">
          <span>综合评分</span>
          <span class="font-weight-medium">{{ scorePercent }}%</span>
        </div>
        <v-progress-linear
          :model-value="scorePercent"
          color="primary"
          height="6"
          rounded
        />
      </div>

      <!-- Statistics -->
      <div class="d-flex ga-2 flex-wrap">
        <v-chip size="x-small" variant="tonal" color="primary">
          <v-icon start size="12">mdi-target</v-icon>
          Top1: {{ top1Percent }}%
        </v-chip>
        <v-chip size="x-small" variant="tonal" color="secondary">
          <v-icon start size="12">mdi-counter</v-icon>
          命中: {{ candidate.hits }}
        </v-chip>
        <v-chip
          v-if="candidate.top_page_no !== null"
          size="x-small"
          variant="tonal"
          color="tertiary"
        >
          <v-icon start size="12">mdi-file-image</v-icon>
          页: {{ candidate.top_page_no }}
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
