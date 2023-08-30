import Vue from 'vue'
import VueRouter from 'vue-router'
import HomeView from '../views/HomeView.vue'
import app from '../views/app.vue'
import Linear from '../views/Linear.vue'

import M1 from '../views/m1.vue'
import M2 from '../views/m2.vue'
import M3 from '../views/m3.vue'
import M4 from '../views/m4.vue'

import View from '../views/view.vue'

Vue.use(VueRouter)

const routes = [
  {
    path: '/',
    name: 'app',
    component: app
  },
  {
    path: '/about',
    name: 'about',
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () => import(/* webpackChunkName: "about" */ '../views/AboutView.vue')
  },
  {
    path: '/linear',
    name: 'Linear',
    component: Linear
  },

  {
    path: '/m1',
    name: 'M1',
    component: M1
  },

  {
    path: '/m2',
    name: 'M2',
    component: M2
  },

  {
    path: '/m3',
    name: 'M3',
    component: M3
  },
  {
    path: '/m4',
    name: 'M4',
    component: M4
  },
  {
    path: '/view',
    name: 'View',
    component: View
  },
]

const router = new VueRouter({
  routes
})

export default router
