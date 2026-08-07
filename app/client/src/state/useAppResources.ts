import { useContext } from 'react'
import { AppResourcesContext, type AppResources } from './appResourcesContext'

export function useAppResources(): AppResources {
  const value = useContext(AppResourcesContext)
  if (!value) throw new Error('useAppResources must be used within AppResourcesProvider')
  return value
}
