/**
 * Application entry point.
 * Registers Vue, Pinia, Router, and Vuetify.
 * Restores auth token from localStorage synchronously BEFORE mounting
 * so that protected page API calls on mount already carry the Bearer token.
 */
import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'
import vuetify from './plugins/vuetify'
import { useAuthStore } from './stores/auth'
import { setAuthToken } from './api/client'

import './styles/main.scss'

const TOKEN_KEY = 'comicsearch-token'

// ── Restore auth token synchronously before the app mounts ──
// This ensures the very first API call from any page component already
// carries the Authorization header, preventing a 401 → login redirect
// while the async checkAuthStatus() is still in flight.
const storedToken = localStorage.getItem(TOKEN_KEY)
if (storedToken) {
    setAuthToken(storedToken)
}

const app = createApp(App)

const pinia = createPinia()
app.use(pinia)
app.use(router)
app.use(vuetify)

// Kick off async auth validation (non-blocking — the sync token above
// covers the gap until this resolves).
const authStore = useAuthStore()
authStore.checkAuthStatus().then(() => {
    if (authStore.loggedIn) {
        authStore.fetchQuota()
    }
})

app.mount('#app')
