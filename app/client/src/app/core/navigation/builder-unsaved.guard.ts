// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import type { CanDeactivateFn } from '@angular/router'
import type { ModelBuilderPageComponent } from '../../features/builder/model-builder-page.component'

export const builderUnsavedGuard: CanDeactivateFn<ModelBuilderPageComponent> = (component, _currentRoute, _currentState, nextState) => component.canDeactivate(nextState?.url ?? '')
