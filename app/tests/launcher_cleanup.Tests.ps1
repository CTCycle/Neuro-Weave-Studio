$cleanupModule = Join-Path $PSScriptRoot '..\scripts\launcher_cleanup.psm1'
Import-Module -Name $cleanupModule -Force

function New-CleanupFixture {
    $base = Join-Path $env:TEMP ("nw-cleanup-{0}" -f [guid]::NewGuid().ToString('N'))
    $repo = Join-Path $base 'repo'
    $resource = Join-Path $repo 'data'
    $client = Join-Path $repo 'app\client'
    $server = Join-Path $repo 'app\server'
    $scripts = Join-Path $repo 'app\scripts'
    $tests = Join-Path $repo 'app\tests'
    $runtime = Join-Path $repo 'runtimes'
    $cache = Join-Path $runtime 'cache'
    foreach ($directory in @($client, $server, $scripts, $tests, $resource, $runtime, $cache)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    return @{
        Base = $base
        RepoRoot = $repo
        ResourceRoot = $resource
        RuntimeRoot = $runtime
        CacheRoot = $cache
        PythonDir = Join-Path $runtime 'python'
        UvDir = Join-Path $runtime 'uv'
        NodeDir = Join-Path $runtime 'node'
        UvVersion = '0.12.0'
        NodeVersion = '22.23.1'
        VenvDir = Join-Path $server '.venv'
        ClientDir = $client
        ServerDir = $server
        ScriptsDir = $scripts
        NodeModulesDir = Join-Path $client 'node_modules'
        LogDir = Join-Path $resource 'runtime\logs'
        ConfiguredLogFile = Join-Path $resource 'runtime\logs\backend.log'
        ConfiguredJobRoot = Join-Path $resource 'runtime\ml-jobs'
        PlaywrightResults = Join-Path $repo 'assets\QA\playwright\test-results'
        BackendInstallState = Join-Path $resource 'runtime\backend-installation.json'
        DatabaseConfigPath = Join-Path $resource 'database-config.json'
        EnvironmentFile = Join-Path $repo 'settings\.env'
        UvArchive = Join-Path $runtime 'uv\uv-0.12.0.zip'
        NodeArchive = Join-Path $runtime 'node\node-v22.23.1-win-x64.zip'
        DataTargets = @()
    }
}

function Write-CleanupFixtureFile {
    param([string]$Path, [string]$Text = 'fixture')
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    [IO.File]::WriteAllText($Path, $Text)
}

Describe 'NeuroWeave launcher cleanup' {
    BeforeEach {
        $script:Fixture = New-CleanupFixture
        $script:Paths = $script:Fixture
    }

    AfterEach {
        if ($script:Fixture -and (Test-Path -LiteralPath $script:Fixture.Base)) {
            Get-ChildItem -LiteralPath $script:Fixture.Base -Force -Recurse -ErrorAction SilentlyContinue |
                Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 } |
                ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue }
            Remove-Item -LiteralPath $script:Fixture.Base -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It 'previews nested hidden caches, removes them repeatedly, and preserves data and dependencies' {
        $ruff = Join-Path $script:Fixture.RepoRoot '.ruff_cache\deep\entry'
        $bytecode = Join-Path $script:Fixture.ServerDir 'package\__pycache__\module.pyc'
        $vite = Join-Path $script:Fixture.NodeModulesDir '.vite\vitest\cache'
        $clientCache = Join-Path $script:Fixture.NodeModulesDir '.cache\tool\entry'
        $angular = Join-Path $script:Fixture.ClientDir '.angular\cache\entry'
        $results = Join-Path $script:Fixture.PlaywrightResults 'run\trace.zip'
        $coverage = Join-Path $script:Fixture.ClientDir 'coverage\lcov.info'
        $coverageShard = Join-Path $script:Fixture.ServerDir '.coverage.integration'
        $tsBuildInfo = Join-Path $script:Fixture.ClientDir 'tsconfig.generated.tsbuildinfo'
        $dependency = Join-Path $script:Fixture.NodeModulesDir 'package\index.js'
        $database = Join-Path $script:Fixture.ResourceRoot 'database.db'
        $envFile = $script:Fixture.EnvironmentFile
        foreach ($path in @($ruff, $bytecode, $vite, $clientCache, $angular, $results, $coverage, $coverageShard, $tsBuildInfo, $dependency, $database, $envFile)) {
            Write-CleanupFixtureFile -Path $path
        }

        $plan = New-NwCleanupPlan -Operation ClearCache -Paths $script:Paths
        $preview = Invoke-NwCleanupPlan -Plan $plan -DryRun
        (@($preview.Planned) -contains [IO.Path]::GetFullPath((Join-Path $script:Fixture.RepoRoot '.ruff_cache'))) | Should Be $true
        (@($preview.Planned) -contains [IO.Path]::GetFullPath((Split-Path -Parent $bytecode))) | Should Be $true
        (@($preview.Planned) -contains [IO.Path]::GetFullPath((Join-Path $script:Fixture.NodeModulesDir '.vite'))) | Should Be $true
        (@($preview.Planned) -contains [IO.Path]::GetFullPath((Join-Path $script:Fixture.ClientDir 'coverage'))) | Should Be $true
        (@($preview.Planned) -contains [IO.Path]::GetFullPath($coverageShard)) | Should Be $true
        (@($preview.Planned) -contains [IO.Path]::GetFullPath($tsBuildInfo)) | Should Be $true
        (Test-Path -LiteralPath $dependency) | Should Be $true
        (Test-Path -LiteralPath $database) | Should Be $true
        (Test-Path -LiteralPath $envFile) | Should Be $true

        $result = Invoke-NwCleanupPlan -Plan $plan
        $result.Failed.Count | Should Be 0
        (Test-Path -LiteralPath $dependency) | Should Be $true
        (Test-Path -LiteralPath $database) | Should Be $true
        (Test-Path -LiteralPath $envFile) | Should Be $true
        (Test-Path -LiteralPath $script:Fixture.PlaywrightResults) | Should Be $false

        $again = New-NwCleanupPlan -Operation ClearCache -Paths $script:Paths
        $repeat = Invoke-NwCleanupPlan -Plan $again
        $repeat.Failed.Count | Should Be 0
    }

    It 'standard uninstall removes managed components and logs but keeps persistent data' {
        $venvFile = Join-Path $script:Fixture.VenvDir 'Scripts\python.exe'
        $runtimeFile = Join-Path $script:Fixture.PythonDir 'python.exe'
        $nodeFile = Join-Path $script:Fixture.NodeModulesDir 'package\index.js'
        $logFile = $script:Fixture.ConfiguredLogFile
        $workerLog = Join-Path $script:Fixture.ConfiguredJobRoot 'job-1\worker.log'
        $jobCheckpoint = Join-Path $script:Fixture.ConfiguredJobRoot 'job-1\checkpoint.bin'
        $stateTemporary = Join-Path (Split-Path -Parent $script:Fixture.BackendInstallState) '.backend-installation.json.0123456789abcdef0123456789abcdef.tmp'
        $legacyStateTemporary = [IO.Path]::ChangeExtension($script:Fixture.BackendInstallState, '.tmp')
        $database = Join-Path $script:Fixture.ResourceRoot 'database.db'
        $export = Join-Path $script:Fixture.ResourceRoot 'runtime\exports\model.onnx'
        $envFile = $script:Fixture.EnvironmentFile
        foreach ($path in @($venvFile, $runtimeFile, $nodeFile, $logFile, $workerLog, $jobCheckpoint, $stateTemporary, $legacyStateTemporary, $database, $export, $envFile)) {
            Write-CleanupFixtureFile -Path $path
        }

        $plan = New-NwCleanupPlan -Operation Uninstall -Paths $script:Paths
        $result = Invoke-NwCleanupPlan -Plan $plan
        $result.Failed.Count | Should Be 0
        (Test-Path -LiteralPath $script:Fixture.VenvDir) | Should Be $false
        (Test-Path -LiteralPath $script:Fixture.NodeModulesDir) | Should Be $false
        (Test-Path -LiteralPath $logFile) | Should Be $false
        (Test-Path -LiteralPath $workerLog) | Should Be $false
        (Test-Path -LiteralPath $script:Fixture.ConfiguredJobRoot) | Should Be $true
        (Test-Path -LiteralPath $jobCheckpoint) | Should Be $true
        (Test-Path -LiteralPath $stateTemporary) | Should Be $false
        (Test-Path -LiteralPath $legacyStateTemporary) | Should Be $false
        (Test-Path -LiteralPath $database) | Should Be $true
        (Test-Path -LiteralPath $export) | Should Be $true
        (Test-Path -LiteralPath $envFile) | Should Be $true
    }

    It 'complete data removal deletes only listed app roots and leaves neighboring files intact' {
        $envFile = $script:Fixture.EnvironmentFile
        $config = $script:Fixture.DatabaseConfigPath
        $configTemporary = Join-Path $script:Fixture.ResourceRoot '.database-config.json.0123456789.tmp'
        $database = Join-Path $script:Fixture.ResourceRoot 'database.db'
        $runtime = Join-Path $script:Fixture.ResourceRoot 'runtime\ml-jobs\job\checkpoint.bin'
        $externalRoot = Join-Path $script:Fixture.Base 'selected-exports'
        $externalFile = Join-Path $externalRoot 'model.onnx'
        $configuredNestedOutput = Join-Path $script:Fixture.ResourceRoot 'runtime\exports'
        $neighbor = Join-Path $script:Fixture.Base 'keep.txt'
        foreach ($path in @($envFile, $config, $configTemporary, $database, $runtime, $externalFile, $neighbor)) { Write-CleanupFixtureFile -Path $path }
        $script:Paths.DataTargets = @(
            [pscustomobject]@{ Path = $envFile; Scope = 'Repository'; Kind = 'File'; Reason = 'Local settings' },
            [pscustomobject]@{ Path = $config; Scope = 'Resource'; Kind = 'File'; Reason = 'DB configuration' },
            [pscustomobject]@{ Path = $configTemporary; Scope = 'Resource'; Kind = 'File'; Reason = 'abandoned DB config temp' },
            [pscustomobject]@{ Path = $database; Scope = 'Resource'; Kind = 'File'; Reason = 'DB' },
            [pscustomobject]@{ Path = (Join-Path $script:Fixture.ResourceRoot 'runtime'); Scope = 'Resource'; Kind = 'Directory'; Reason = 'Runtime data' },
            [pscustomobject]@{ Path = $configuredNestedOutput; Scope = 'Resource'; Kind = 'Directory'; Reason = 'Configured export output' },
            [pscustomobject]@{ Path = $externalRoot; Scope = 'Configured'; Kind = 'Directory'; Reason = 'Configured exports' }
        )

        $plan = New-NwCleanupPlan -Operation RemoveAllData -Paths $script:Paths
        $preview = Invoke-NwCleanupPlan -Plan $plan -DryRun
        (@($preview.Planned) -contains [IO.Path]::GetFullPath($envFile)) | Should Be $true
        (@($preview.Planned) -contains [IO.Path]::GetFullPath($configuredNestedOutput)) | Should Be $true
        (@($preview.Planned) -contains [IO.Path]::GetFullPath($externalRoot)) | Should Be $true
        (@($preview.Planned) -contains [IO.Path]::GetFullPath($configTemporary)) | Should Be $true
        $result = Invoke-NwCleanupPlan -Plan $plan
        $result.Failed.Count | Should Be 0
        (Test-Path -LiteralPath $envFile) | Should Be $false
        (Test-Path -LiteralPath $database) | Should Be $false
        (Test-Path -LiteralPath $configTemporary) | Should Be $false
        (Test-Path -LiteralPath $externalRoot) | Should Be $false
        (Test-Path -LiteralPath $neighbor) | Should Be $true
    }

    It 'rejects a broad user-profile target and does not traverse a junction' {
        $broadPlan = [pscustomobject]@{
            Operation = 'RemoveAllData'
            RepositoryRoot = $script:Fixture.RepoRoot
            ResourceRoot = $script:Fixture.ResourceRoot
            DiscoveryFailures = @()
            Targets = @([pscustomobject]@{ Path = $env:USERPROFILE; Scope = 'Configured'; Kind = 'Directory'; Reason = 'invalid broad target' })
        }
        $broad = Invoke-NwCleanupPlan -Plan $broadPlan -DryRun
        $broad.Failed.Count | Should BeGreaterThan 0

        $broadResourcePlan = [pscustomobject]@{
            Operation = 'RemoveAllData'
            RepositoryRoot = $script:Fixture.RepoRoot
            ResourceRoot = $script:Fixture.RepoRoot
            DiscoveryFailures = @()
            Targets = @([pscustomobject]@{ Path = (Join-Path $script:Fixture.RepoRoot 'data\runtime'); Scope = 'Resource'; Kind = 'Directory'; Reason = 'broad resource root' })
        }
        $broadResource = Invoke-NwCleanupPlan -Plan $broadResourcePlan -DryRun
        $broadResource.Failed.Count | Should BeGreaterThan 0

        $programFiles = [Environment]::GetEnvironmentVariable('ProgramFiles')
        if (-not [string]::IsNullOrWhiteSpace($programFiles)) {
            $systemPlan = [pscustomobject]@{
                Operation = 'RemoveAllData'
                RepositoryRoot = $script:Fixture.RepoRoot
                ResourceRoot = $script:Fixture.ResourceRoot
                DiscoveryFailures = @()
                Targets = @([pscustomobject]@{ Path = (Join-Path $programFiles 'NeuroWeave-test-output'); Scope = 'Configured'; Kind = 'Directory'; Reason = 'protected system location' })
            }
            $systemTarget = Invoke-NwCleanupPlan -Plan $systemPlan -DryRun
            $systemTarget.Failed.Count | Should BeGreaterThan 0
        }

        $outside = Join-Path $script:Fixture.Base 'outside'
        $sentinel = Join-Path $outside 'keep.txt'
        Write-CleanupFixtureFile -Path $sentinel
        $junction = $script:Fixture.CacheRoot
        New-Item -ItemType Junction -Path $junction -Target $outside -ErrorAction Stop | Out-Null
        $linkPlan = New-NwCleanupPlan -Operation ClearCache -Paths $script:Paths
        $linkResult = Invoke-NwCleanupPlan -Plan $linkPlan
        $linkResult.Failed.Count | Should BeGreaterThan 0
        (Test-Path -LiteralPath $sentinel) | Should Be $true
    }

    It 'reports a locked file while continuing to remove other targets' {
        $locked = Join-Path $script:Fixture.ResourceRoot 'runtime\locked.log'
        $available = Join-Path $script:Fixture.ResourceRoot 'runtime\available.log'
        Write-CleanupFixtureFile -Path $locked
        Write-CleanupFixtureFile -Path $available
        $handle = [IO.File]::Open($locked, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
        try {
            $plan = [pscustomobject]@{
                Operation = 'CleanLogs'
                RepositoryRoot = $script:Fixture.RepoRoot
                ResourceRoot = $script:Fixture.ResourceRoot
                DiscoveryFailures = @()
                Targets = @(
                    [pscustomobject]@{ Path = $locked; Scope = 'Resource'; Kind = 'File'; Reason = 'locked' },
                    [pscustomobject]@{ Path = $available; Scope = 'Resource'; Kind = 'File'; Reason = 'available' }
                )
            }
            $result = Invoke-NwCleanupPlan -Plan $plan
            $result.Failed.Count | Should BeGreaterThan 0
            (Test-Path -LiteralPath $locked) | Should Be $true
            (Test-Path -LiteralPath $available) | Should Be $false
        } finally { $handle.Dispose() }
    }

    It 'reports an access-denied deletion and continues other targets' {
        $denied = Join-Path $script:Fixture.ResourceRoot 'runtime\denied.log'
        $available = Join-Path $script:Fixture.ResourceRoot 'runtime\available.log'
        Write-CleanupFixtureFile -Path $denied
        Write-CleanupFixtureFile -Path $available
        Mock -CommandName Remove-Item -MockWith { throw [UnauthorizedAccessException]::new('access denied') } -ParameterFilter { $LiteralPath -like '*denied.log' } -ModuleName launcher_cleanup
        $plan = [pscustomobject]@{
            Operation = 'CleanLogs'
            RepositoryRoot = $script:Fixture.RepoRoot
            ResourceRoot = $script:Fixture.ResourceRoot
            DiscoveryFailures = @()
            DiscoverySkipped = @()
            Targets = @(
                [pscustomobject]@{ Path = $denied; Scope = 'Resource'; Kind = 'File'; Reason = 'denied' },
                [pscustomobject]@{ Path = $available; Scope = 'Resource'; Kind = 'File'; Reason = 'available' }
            )
        }
        $result = Invoke-NwCleanupPlan -Plan $plan
        $result.Failed.Count | Should BeGreaterThan 0
        (Test-Path -LiteralPath $denied) | Should Be $true
        (Test-Path -LiteralPath $available) | Should Be $false
    }

    It 'rejects a directory supplied as a file target and preserves its contents' {
        $directoryTarget = Join-Path $script:Fixture.ResourceRoot 'runtime\not-a-file.log'
        $sentinel = Join-Path $directoryTarget 'keep.txt'
        $available = Join-Path $script:Fixture.ResourceRoot 'runtime\available.log'
        Write-CleanupFixtureFile -Path $sentinel
        Write-CleanupFixtureFile -Path $available
        $plan = [pscustomobject]@{
            Operation = 'CleanLogs'
            RepositoryRoot = $script:Fixture.RepoRoot
            ResourceRoot = $script:Fixture.ResourceRoot
            DiscoveryFailures = @()
            Targets = @(
                [pscustomobject]@{ Path = $directoryTarget; Scope = 'Resource'; Kind = 'File'; Reason = 'file expected' },
                [pscustomobject]@{ Path = $available; Scope = 'Resource'; Kind = 'File'; Reason = 'valid file' }
            )
        }
        $preview = Invoke-NwCleanupPlan -Plan $plan -DryRun
        $preview.Failed.Count | Should BeGreaterThan 0
        $result = Invoke-NwCleanupPlan -Plan $plan
        $result.Failed.Count | Should BeGreaterThan 0
        (Test-Path -LiteralPath $sentinel) | Should Be $true
        (Test-Path -LiteralPath $available) | Should Be $false
    }
}
