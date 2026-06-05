import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const getModelCurrent   = () => api.get('/model/current')
export const getAccuracy       = () => api.get('/predictions/accuracy')
export const getAccuracyByWC   = () => api.get('/predictions/accuracy/by-weight-class')
export const getAccuracyByVer  = () => api.get('/predictions/accuracy/by-version')
export const getHistory        = () => api.get('/predictions/history')
export const getUpcoming       = () => api.get('/predictions/upcoming')
export const searchFighters    = (q) => api.get(`/fighters/search?q=${q}`)
export const predict           = (body) => api.post('/predict', body)
