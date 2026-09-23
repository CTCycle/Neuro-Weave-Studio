# NW-T1-05 Codex in-app Browser observation

Date: 2026-09-23
Preview: `http://127.0.0.1:6425`
Isolated run database: `runtimes/cache/comprehensive-validation-20260923-t1-05/neuroweave.db`

The Settings page displayed “There are no editable application-level settings
defined yet.” Its Node Library link targeted `/nodes`, and its Runtime
Environment link targeted `/environment`. The page identified environment
variables, `.env` files, ports, hosts, database paths, logs, job roots, and
launcher configuration as outside the page.

With no project or dataset registered, Training displayed both prerequisite
messages and disabled its draft creation controls. After adding one disposable
saved model through the local project API and registering a two-row CSV through
the dataset UI, Training showed a draft named `NW T1-05 in-app session draft`
with the selected model and dataset, 7 epochs, and batch size 16. Its status
was `DRAFT · not started`, and **Start training** was disabled.

The draft remained after navigating to Settings and back to Training. A full
reload showed “No training sessions.” A screenshot of the equivalent draft
state from the serial browser workflow is included in
[Training draft screenshot](nw-t1-05-training-session-draft.png); the automated
browser run captured the Settings and after-reload states separately.

Model Builder displayed “No operational execution backend is available for a
new model” in this launch, so the isolated project record was created through
the existing local API for this boundary check. The dataset was registered
through the UI. Both records and the CSV were deleted after inspection.
