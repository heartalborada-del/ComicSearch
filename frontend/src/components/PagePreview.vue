<script setup lang="ts">
/**
 * Page preview — displays a matched page image with click-to-zoom dialog.
 * Uses pack_id + page_no to construct the image URL.
 */
import { computed, ref } from 'vue'
import { pageUrl } from '@/utils/image-url'

const props = defineProps<{
  packId: number
  pageNo: number
}>()

const dialog = ref(false)

const imageUrl = computed(() => pageUrl(props.packId, props.pageNo))
</script>

<template>
  <div class="page-preview">
    <v-card
      variant="tonal"
      class="bg-surface-container state-layer cursor-pointer"
      rounded="lg"
      @click="dialog = true"
    >
      <v-img
        :src="imageUrl"
        aspect-ratio="0.7"
        cover
        class="bg-surface-container-high"
      >
        <template #placeholder>
          <div class="d-flex align-center justify-center fill-height">
            <v-progress-circular indeterminate color="primary" size="32" />
          </div>
        </template>
        <template #error>
          <div class="d-flex align-center justify-center fill-height">
            <v-icon size="40" color="outline">mdi-file-image-outline</v-icon>
          </div>
        </template>
      </v-img>
      <div class="text-caption text-center text-on-surface-variant pa-2">
        第 {{ pageNo }} 页
      </div>
    </v-card>

    <!-- Zoom dialog -->
    <v-dialog v-model="dialog" max-width="900" transition="fade-transition">
      <v-card rounded="xl" color="surface">
        <v-toolbar flat class="bg-surface-container">
          <v-toolbar-title class="text-body-1">
            Pack #{{ packId }} — 第 {{ pageNo }} 页
          </v-toolbar-title>
          <v-btn icon="mdi-close" variant="text" @click="dialog = false" />
        </v-toolbar>
        <v-card-text class="d-flex justify-center pa-4">
          <v-img
            :src="imageUrl"
            max-height="80vh"
            contain
          >
            <template #placeholder>
              <div class="d-flex align-center justify-center fill-height">
                <v-progress-circular indeterminate color="primary" size="48" />
              </div>
            </template>
          </v-img>
        </v-card-text>
      </v-card>
    </v-dialog>
  </div>
</template>

<style scoped>
.cursor-pointer {
  cursor: pointer;
}
</style>
