Set-StrictMode -Version Latest
$script:DiscoveryFailures = [Collections.Generic.List[string]]::new()
$script:DiscoverySkipped = [Collections.Generic.List[string]]::new()

function Test-NwPathWithin {
    param([string]$Child, [string]$Parent)
    if ([string]::IsNullOrWhiteSpace($Child) -or [string]::IsNullOrWhiteSpace($Parent)) { return $false }
    $childPath = [IO.Path]::GetFullPath($Child).TrimEnd('\')
    $parentPath = [IO.Path]::GetFullPath($Parent).TrimEnd('\')
    if ($childPath -ieq $parentPath) { return $true }
    return $childPath.StartsWith("$parentPath\", [StringComparison]::OrdinalIgnoreCase)
}

function Add-NwCleanupTarget {
    param(
        [Collections.Generic.List[object]]$Targets,
        [string]$Path,
        [ValidateSet('Repository', 'Resource', 'Configured')][string]$Scope,
        [ValidateSet('File', 'Directory', 'LogFiles')][string]$Kind,
        [string]$Reason
    )
    if ([string]::IsNullOrWhiteSpace($Path)) { return }
    try { $fullPath = [IO.Path]::GetFullPath($Path) } catch { return }
    if (@($Targets | Where-Object { $_.Path -ieq $fullPath }).Count -eq 0) {
        $Targets.Add([pscustomobject]@{ Path = $fullPath; Scope = $Scope; Kind = $Kind; Reason = $Reason })
    }
}

function Get-NwTreeEntries {
    param([string]$Root)
    $found = [Collections.Generic.List[object]]::new()
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) { return @() }
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push([IO.Path]::GetFullPath($Root))
    while ($pending.Count -gt 0) {
        $current = $pending.Pop()
        try {
            $item = Get-Item -LiteralPath $current -Force -ErrorAction Stop
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                $script:DiscoverySkipped.Add("$current (symbolic link or junction; not traversed)")
                continue
            }
            $children = @(Get-ChildItem -LiteralPath $current -Force -ErrorAction Stop)
        } catch {
            $script:DiscoveryFailures.Add("$current ($($_.Exception.Message))")
            continue
        }
        foreach ($child in $children) {
            if (($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                $script:DiscoverySkipped.Add("$($child.FullName) (symbolic link or junction; not traversed)")
                continue
            }
            if ($child.PSIsContainer) {
                if ($child.Name -in @('__pycache__', '.pytest_cache', '.ruff_cache', '.vite', 'htmlcov', 'coverage')) {
                    $found.Add([pscustomobject]@{ Path = $child.FullName; Kind = 'Directory' })
                    continue
                }
                if ($child.Name -in @('.git', 'node_modules', '.venv', 'runtimes', 'assets', 'data', 'settings')) { continue }
                $pending.Push($child.FullName)
            }
            elseif (($child.Name -match '\.py[co]$') -or $child.Name -match '^\.coverage(?:\..+)?$' -or $child.Name -in @('.eslintcache') -or $child.Name -match '\.tsbuildinfo$') {
                $found.Add([pscustomobject]@{ Path = $child.FullName; Kind = 'File' })
            }
        }
    }
    return @($found)
}

function Get-NwLogFiles {
    param([string]$Root, [string[]]$ExactNames = @())
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) { return @() }
    $files = [Collections.Generic.List[string]]::new()
    $pending = [Collections.Generic.Stack[object]]::new()
    $pending.Push([pscustomobject]@{ Path = [IO.Path]::GetFullPath($Root); Depth = 0 })
    while ($pending.Count -gt 0) {
        $entry = $pending.Pop()
        $current = $entry.Path
        try {
            $item = Get-Item -LiteralPath $current -Force -ErrorAction Stop
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                $script:DiscoverySkipped.Add("$current (symbolic link or junction; not traversed)")
                continue
            }
            $children = @(Get-ChildItem -LiteralPath $current -Force -ErrorAction Stop)
        } catch {
            $script:DiscoveryFailures.Add("$current ($($_.Exception.Message))")
            continue
        }
        foreach ($child in $children) {
            if (($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                $script:DiscoverySkipped.Add("$($child.FullName) (symbolic link or junction; not traversed)")
                continue
            }
            if ($child.PSIsContainer) {
                if ($ExactNames.Count -eq 0 -or $entry.Depth -eq 0) {
                    $pending.Push([pscustomobject]@{ Path = $child.FullName; Depth = $entry.Depth + 1 })
                }
            }
            elseif (($ExactNames.Count -gt 0 -and $child.Name -in $ExactNames) -or ($ExactNames.Count -eq 0 -and $child.Extension -ieq '.log')) { $files.Add($child.FullName) }
        }
    }
    return @($files)
}

function Get-NwAdjacentAtomicTemps {
    param([string]$ReferencePath, [string]$Pattern)
    if ([string]::IsNullOrWhiteSpace($ReferencePath)) { return @() }
    $parent = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($ReferencePath))
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) { return @() }
    try {
        $parentItem = Get-Item -LiteralPath $parent -Force -ErrorAction Stop
        if (($parentItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            $script:DiscoverySkipped.Add("$parent (symbolic link or junction; not traversed)")
            return @()
        }
        $items = @(Get-ChildItem -LiteralPath $parent -Force -File -ErrorAction Stop)
    } catch {
        $script:DiscoveryFailures.Add("$parent ($($_.Exception.Message))")
        return @()
    }
    $matchedFiles = [Collections.Generic.List[string]]::new()
    foreach ($item in $items) {
        if ($item.Name -notmatch $Pattern) { continue }
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            $script:DiscoverySkipped.Add("$($item.FullName) (symbolic link or junction; not traversed)")
            continue
        }
        $matchedFiles.Add($item.FullName)
    }
    return @($matchedFiles)
}

function Get-NwFrontendCacheDirectories {
    param([string]$Root)
    $found = [Collections.Generic.List[string]]::new()
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) { return @() }
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push([IO.Path]::GetFullPath($Root))
    while ($pending.Count -gt 0) {
        $current = $pending.Pop()
        try {
            $attributes = [IO.File]::GetAttributes($current)
            if (($attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                $script:DiscoverySkipped.Add("$current (symbolic link or junction; not traversed)")
                continue
            }
            $directories = @([IO.Directory]::EnumerateDirectories($current))
        } catch {
            $script:DiscoveryFailures.Add("$current ($($_.Exception.Message))")
            continue
        }
        foreach ($directory in $directories) {
            try {
                $attributes = [IO.File]::GetAttributes($directory)
                if (($attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                    $script:DiscoverySkipped.Add("$directory (symbolic link or junction; not traversed)")
                    continue
                }
                $name = [IO.Path]::GetFileName($directory)
                if ($name -in @('.cache', '.vite')) {
                    $found.Add($directory)
                    continue
                }
                if ($name -eq '.angular') {
                    $angularCache = Join-Path $directory 'cache'
                    if (Test-Path -LiteralPath $angularCache -PathType Container) { $found.Add($angularCache) }
                    continue
                }
                $pending.Push($directory)
            } catch { $script:DiscoveryFailures.Add("$directory ($($_.Exception.Message))") }
        }
    }
    return @($found)
}

function New-NwCleanupPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][ValidateSet('ClearCache', 'CleanLogs', 'Uninstall', 'RemoveAllData')][string]$Operation,
        [Parameter(Mandatory = $true)][hashtable]$Paths
    )

    $script:DiscoveryFailures.Clear()
    $script:DiscoverySkipped.Clear()
    $targets = [Collections.Generic.List[object]]::new()
    $repo = [IO.Path]::GetFullPath($Paths.RepoRoot)
    $resource = [IO.Path]::GetFullPath($Paths.ResourceRoot)
    if ($Operation -in @('ClearCache', 'Uninstall')) {
        Add-NwCleanupTarget $targets $Paths.CacheRoot Repository Directory 'Canonical runtime, tooling, test, and frontend build cache'
        Add-NwCleanupTarget $targets $Paths.PlaywrightResults Repository Directory 'Generated Playwright test results'
        Add-NwCleanupTarget $targets $Paths.UvArchive Repository File 'Interrupted uv download archive'
        Add-NwCleanupTarget $targets $Paths.NodeArchive Repository File 'Interrupted Node.js download archive'

        if (Test-Path -LiteralPath $repo -PathType Container) {
            foreach ($entry in @(Get-NwTreeEntries -Root $repo)) {
                if (Test-NwPathWithin -Child $entry.Path -Parent $Paths.CacheRoot) { continue }
                if ($entry.Kind -eq 'Directory' -and $entry.Path -match '[\\/]node_modules([\\/]|$)') { continue }
                Add-NwCleanupTarget $targets $entry.Path Repository $entry.Kind 'Recognized source-tree cache or bytecode'
            }
        }

        if ($Operation -eq 'ClearCache') {
            foreach ($directory in @(Get-NwFrontendCacheDirectories -Root $Paths.NodeModulesDir)) {
                Add-NwCleanupTarget $targets $directory Repository Directory 'Nested frontend dependency cache'
            }
        }

        foreach ($candidate in @(
                (Join-Path $Paths.ClientDir 'node_modules/.cache'),
                (Join-Path $Paths.ClientDir 'node_modules/.vite'),
                (Join-Path $Paths.ClientDir '.vite'),
                (Join-Path $Paths.ClientDir '.angular/cache')
            )) {
            Add-NwCleanupTarget $targets $candidate Repository Directory 'Recognized frontend cache'
        }
        foreach ($candidate in @(
                (Join-Path $Paths.UvDir ("uv-$($Paths.UvVersion).zip")),
                (Join-Path $Paths.NodeDir ("node-v$($Paths.NodeVersion)-win-x64.zip"))
            )) {
            Add-NwCleanupTarget $targets $candidate Repository File 'Interrupted runtime download archive'
        }
    }

    if ($Operation -in @('CleanLogs', 'Uninstall')) {
        foreach ($file in @(Get-NwLogFiles -Root $Paths.LogDir)) {
            Add-NwCleanupTarget $targets $file Resource File 'Application-generated log'
        }
        if ($Paths.ConfiguredLogFile) {
            Add-NwCleanupTarget $targets $Paths.ConfiguredLogFile Configured File 'Database-configured application log file'
        }
        if ($Paths.ConfiguredJobRoot) {
            $jobRoot = [IO.Path]::GetFullPath($Paths.ConfiguredJobRoot)
            $jobScope = if (Test-NwPathWithin -Child $jobRoot -Parent $resource) { 'Resource' } elseif (Test-NwPathWithin -Child $jobRoot -Parent $repo) { 'Repository' } else { 'Configured' }
            try {
                [void](Assert-NwCleanupTarget ([pscustomobject]@{ Path = $jobRoot; Scope = $jobScope; Kind = 'Directory' }) $repo $resource)
                foreach ($file in @(Get-NwLogFiles -Root $jobRoot -ExactNames @('worker.log'))) {
                    Add-NwCleanupTarget $targets $file $jobScope File 'Per-job application worker log'
                }
            } catch { $script:DiscoveryFailures.Add("$jobRoot (worker logs not scanned: $($_.Exception.Message))") }
        }
    }

    if ($Operation -eq 'Uninstall') {
        foreach ($candidate in @($Paths.PythonDir, $Paths.UvDir, $Paths.NodeDir, $Paths.CacheRoot, $Paths.VenvDir, $Paths.NodeModulesDir, $Paths.PlaywrightResults)) {
            Add-NwCleanupTarget $targets $candidate Repository Directory 'Managed installation component'
        }
        Add-NwCleanupTarget $targets $Paths.BackendInstallState Resource File 'Launcher backend installation state'
        $stateName = [regex]::Escape([IO.Path]::GetFileName($Paths.BackendInstallState))
        foreach ($temporary in @(Get-NwAdjacentAtomicTemps -ReferencePath $Paths.BackendInstallState -Pattern "^\.$stateName\.[0-9a-f]{32}\.(?:tmp|bak)$")) {
            Add-NwCleanupTarget $targets $temporary Resource File 'Abandoned launcher installation-state temporary'
        }
        Add-NwCleanupTarget $targets ([IO.Path]::ChangeExtension($Paths.BackendInstallState, '.tmp')) Resource File 'Abandoned backend installation-state temporary'
    }

    if ($Operation -eq 'RemoveAllData') {
        foreach ($entry in @($Paths.DataTargets)) {
            if ($null -ne $entry) { Add-NwCleanupTarget $targets $entry.Path $entry.Scope $entry.Kind $entry.Reason }
        }
        $databaseConfigName = [regex]::Escape([IO.Path]::GetFileName($Paths.DatabaseConfigPath))
        foreach ($temporary in @(Get-NwAdjacentAtomicTemps -ReferencePath $Paths.DatabaseConfigPath -Pattern "^\.$databaseConfigName\..+\.tmp$")) {
            Add-NwCleanupTarget $targets $temporary Resource File 'Abandoned database-configuration temporary'
        }
    }

    $requested = @($targets)
    # Parent targets subsume their descendants for deletion. Retain the original
    # targets separately so the launcher can preview every configured data path.
    # application-owned root so a path is never deleted twice.
    $ordered = @($targets | Sort-Object @{ Expression = { $_.Path.Length }; Descending = $false }, @{ Expression = { $_.Path.ToUpperInvariant() }; Descending = $false })
    $filtered = [Collections.Generic.List[object]]::new()
    foreach ($target in $ordered) {
        $covered = $false
        foreach ($prior in $filtered) {
            if ($prior.Kind -eq 'Directory' -and (Test-NwPathWithin -Child $target.Path -Parent $prior.Path)) { $covered = $true; break }
        }
        if (-not $covered) { $filtered.Add($target) }
    }
    return [pscustomobject]@{ Operation = $Operation; Targets = @($filtered); RequestedTargets = $requested; RepositoryRoot = $repo; ResourceRoot = $resource; DiscoveryFailures = @($script:DiscoveryFailures); DiscoverySkipped = @($script:DiscoverySkipped) }
}

function Test-NwReparseAncestor {
    param([string]$Path)
    $current = [IO.Path]::GetFullPath($Path)
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            try {
                $item = Get-Item -LiteralPath $current -Force -ErrorAction Stop
                if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { return $current }
            } catch { return $current }
        }
        $parent = [IO.Path]::GetDirectoryName($current.TrimEnd('\'))
        if (-not $parent -or $parent -ieq $current) { break }
        $current = $parent
    }
    return $null
}

function Assert-NwCleanupTarget {
    param([object]$Target, [string]$RepositoryRoot, [string]$ResourceRoot)
    if ([string]::IsNullOrWhiteSpace($Target.Path)) { throw 'Refusing an empty cleanup target.' }
    if ($Target.Kind -notin @('File', 'Directory')) { throw "Refusing an unsupported target kind '$($Target.Kind)' for '$($Target.Path)'." }
    $full = [IO.Path]::GetFullPath($Target.Path)
    if ([IO.Path]::GetPathRoot($full) -ieq $full) { throw "Refusing filesystem root '$full'." }
    if ($Target.Scope -eq 'Repository' -and (-not (Test-NwPathWithin $full $RepositoryRoot) -or $full -ieq [IO.Path]::GetFullPath($RepositoryRoot))) {
        throw "Repository target is outside its allowed root: $full"
    }
    if ($Target.Scope -eq 'Resource' -and (-not (Test-NwPathWithin $full $ResourceRoot) -or $full -ieq [IO.Path]::GetFullPath($ResourceRoot))) {
        throw "Resource target is outside its allowed root: $full"
    }
    if ($Target.Scope -eq 'Resource') {
        $profileRoots = @($env:USERPROFILE, $env:LOCALAPPDATA, $env:APPDATA)
        if ($env:USERPROFILE) {
            foreach ($folder in @('Desktop', 'Documents', 'Downloads', 'Pictures', 'Music', 'Videos')) {
                $profileRoots += Join-Path $env:USERPROFILE $folder
            }
        }
        $systemRoots = @($env:WINDIR, [Environment]::GetEnvironmentVariable('ProgramFiles'), [Environment]::GetEnvironmentVariable('ProgramFiles(x86)'), $env:ProgramData, $env:PUBLIC) | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }
        foreach ($protected in @([IO.Path]::GetPathRoot($ResourceRoot), $RepositoryRoot) + $systemRoots + $profileRoots) {
            if ([string]::IsNullOrWhiteSpace([string]$protected)) { continue }
            $protectedPath = [IO.Path]::GetFullPath($protected)
            if ($ResourceRoot -ieq $protectedPath -or (Test-NwPathWithin -Child $protectedPath -Parent $ResourceRoot)) {
                throw "Configured resource root is too broad or protected: $ResourceRoot"
            }
        }
        foreach ($systemRoot in $systemRoots) {
            if (Test-NwPathWithin -Child $ResourceRoot -Parent ([IO.Path]::GetFullPath($systemRoot))) {
                throw "Configured resource root is inside a protected system location: $ResourceRoot"
            }
        }
    }
    if ($Target.Scope -eq 'Configured') {
        $profileRoots = @($env:USERPROFILE, $env:LOCALAPPDATA, $env:APPDATA)
        if ($env:USERPROFILE) {
            foreach ($folder in @('Desktop', 'Documents', 'Downloads', 'Pictures', 'Music', 'Videos')) {
                $profileRoots += Join-Path $env:USERPROFILE $folder
            }
        }
        $systemRoots = @($env:WINDIR, [Environment]::GetEnvironmentVariable('ProgramFiles'), [Environment]::GetEnvironmentVariable('ProgramFiles(x86)'), $env:ProgramData, $env:PUBLIC) | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }
        foreach ($protected in @($RepositoryRoot, $ResourceRoot) + $systemRoots + $profileRoots) {
            if ([string]::IsNullOrWhiteSpace([string]$protected)) { continue }
            $protectedPath = [IO.Path]::GetFullPath($protected)
            if ($full -ieq $protectedPath -or (Test-NwPathWithin -Child $protectedPath -Parent $full)) {
                throw "Configured target is a protected root or contains one: $full"
            }
        }
        foreach ($systemRoot in $systemRoots) {
            if (Test-NwPathWithin -Child $full -Parent ([IO.Path]::GetFullPath($systemRoot))) {
                throw "Configured target is inside a protected system location: $full"
            }
        }
    }
    $link = Test-NwReparseAncestor -Path $full
    if ($link) { throw "Reparse point in cleanup path; refusing to traverse '$link'." }
    if (Test-Path -LiteralPath $full) {
        try { $item = Get-Item -LiteralPath $full -Force -ErrorAction Stop }
        catch { throw "Could not inspect cleanup target '$full': $($_.Exception.Message)" }
        if ($Target.Kind -eq 'File' -and $item.PSIsContainer) { throw "Expected a file but found a directory at '$full'." }
        if ($Target.Kind -eq 'Directory' -and -not $item.PSIsContainer) { throw "Expected a directory but found a file at '$full'." }
    }
    return $full
}

function Remove-NwTargetSafely {
    param([string]$Path)
    $removedCount = 0
    $skipped = [Collections.Generic.List[string]]::new()
    $failed = [Collections.Generic.List[string]]::new()
    $root = [IO.Path]::GetFullPath($Path)
    $stack = [Collections.Generic.Stack[object]]::new()
    $stack.Push([pscustomobject]@{ Path = $root; Visited = $false })
    while ($stack.Count -gt 0) {
        $entry = $stack.Pop()
        if (-not (Test-NwPathWithin -Child $entry.Path -Parent $root)) {
            $failed.Add("$($entry.Path) (escaped target root)"); continue
        }
        if (-not (Test-Path -LiteralPath $entry.Path)) { $skipped.Add("$($entry.Path) (not present)"); continue }
        try { $item = Get-Item -LiteralPath $entry.Path -Force -ErrorAction Stop }
        catch { $failed.Add("$($entry.Path) ($($_.Exception.Message))"); continue }
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            $skipped.Add("$($entry.Path) (symbolic link or junction)"); continue
        }
        if ($item.PSIsContainer -and -not $entry.Visited) {
            try { $children = @(Get-ChildItem -LiteralPath $entry.Path -Force -ErrorAction Stop) }
            catch { $failed.Add("$($entry.Path) ($($_.Exception.Message))"); continue }
            $stack.Push([pscustomobject]@{ Path = $entry.Path; Visited = $true })
            foreach ($child in $children) { $stack.Push([pscustomobject]@{ Path = $child.FullName; Visited = $false }) }
            continue
        }
        try {
            Remove-Item -LiteralPath $entry.Path -Force -Confirm:$false -ErrorAction Stop
            $removedCount++
        } catch { $failed.Add("$($entry.Path) ($($_.Exception.Message))") }
    }
    return [pscustomobject]@{ RemovedCount = $removedCount; Skipped = @($skipped); Failed = @($failed) }
}

function Invoke-NwCleanupPlan {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][object]$Plan, [switch]$DryRun)
    $planned = [Collections.Generic.List[string]]::new()
    $removedCount = 0
    $skipped = [Collections.Generic.List[string]]::new()
    $failed = [Collections.Generic.List[string]]::new()
    $targetResults = [Collections.Generic.List[object]]::new()
    if ($Plan.PSObject.Properties['DiscoveryFailures']) {
        foreach ($failure in @($Plan.DiscoveryFailures)) { $failed.Add($failure) }
    }
    if ($Plan.PSObject.Properties['DiscoverySkipped']) {
        foreach ($item in @($Plan.DiscoverySkipped)) { $skipped.Add($item) }
    }
    $targetsToValidate = if ($DryRun -and $Plan.PSObject.Properties['RequestedTargets']) { @($Plan.RequestedTargets) } else { @($Plan.Targets) }
    foreach ($target in $targetsToValidate) {
        try { $full = Assert-NwCleanupTarget $target $Plan.RepositoryRoot $Plan.ResourceRoot }
        catch {
            $message = "$($target.Path) ($($_.Exception.Message))"
            $failed.Add($message)
            $targetResults.Add([pscustomobject]@{ Path = $target.Path; RemovedCount = 0; Skipped = @(); Failed = @($message) })
            continue
        }
        $planned.Add($full)
        if ($DryRun) {
            $targetResults.Add([pscustomobject]@{ Path = $full; RemovedCount = 0; Skipped = @(); Failed = @(); DryRun = $true })
            continue
        }
        $result = Remove-NwTargetSafely -Path $full
        $removedCount += $result.RemovedCount
        foreach ($path in $result.Skipped) { $skipped.Add($path) }
        foreach ($path in $result.Failed) { $failed.Add($path) }
        $targetResults.Add([pscustomobject]@{ Path = $full; RemovedCount = $result.RemovedCount; Skipped = @($result.Skipped); Failed = @($result.Failed) })
    }
    return [pscustomobject]@{ Operation = $Plan.Operation; Planned = @($planned); RemovedCount = $removedCount; Skipped = @($skipped); Failed = @($failed); TargetResults = @($targetResults); DryRun = [bool]$DryRun }
}

Export-ModuleMember -Function New-NwCleanupPlan, Invoke-NwCleanupPlan, Test-NwPathWithin
