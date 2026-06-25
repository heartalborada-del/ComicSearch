/**
 * Application entry point.
 * Registers Vue, Pinia, Router, and Vuetify.
 * Initializes auth store for token restoration.
 */
import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'
import vuetify from './plugins/vuetify'
import { useAuthStore } from './stores/auth'

import './styles/main.scss'

const app = createApp(App)

const pinia = createPinia()
app.use(pinia)
app.use(router)
app.use(vuetify)

// Initialize auth store — try to restore session from localStorage
const authStore = useAuthStore()
authStore.checkAuthStatus().then(() => {
    if (authStore.loggedIn) {
        authStore.fetchQuota()
    }
})

app.mount('#app')
