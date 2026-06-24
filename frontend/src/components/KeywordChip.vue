<script setup lang="ts">
/**
 * Keyword chip — displays a keyword as a tonal M3 chip.
 */
import type { Keyword } from '@/types/info'

defineProps<{
  keyword: Keyword
}>()
</script>

<template>
  <v-chip variant="tonal" size="small" color="secondary" label>
    {{ keyword.name }}
  </v-chip>
</template>
