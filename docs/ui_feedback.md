# SDBench Physician UI Review

## Pain Points

The current demo UI exposes only the bare minimum functionality. Issues noted during internal testing include:

- **Unclear workflow.** Early prototypes required prefixing the message with `test:` to order labs. The interface now provides an action dropdown and searchable test list.
- **Minimal feedback.** Failures to load the case or backend errors appear only as browser alerts or silent log entries.
- **Styling shortcomings.** The two-column grid is functional but lacks responsive design, accessible color contrast, or clear delineation between panels.
- **No session status.** Users cannot easily tell which account is logged in or log out without refreshing the page.

## Clinician Feedback

Three clinicians tried the demo and echoed the concerns above. In particular they mentioned confusion around how to request labs and when the conversation was considered complete. They also asked for clearer cost breakdowns and an option to expand the case summary for reference while chatting.

## Proposed UX Outcomes

- **Clarity.** Provide inline hints or a help panel explaining the Ask Question, Order Test, and Provide Diagnosis controls. Include the current username and a logout button.
- **Accessibility.** Adopt higher-contrast styling and ensure the layout works on tablets. Screen reader labels should be added to form controls.
- **Visual update.** Move to a modern component framework (e.g., React with a UI library) and add collapsible panels so users can focus on the chat while still referencing test results or the summary.

These improvements aim to make SDBench more approachable for clinicians evaluating the Gatekeeper approach.
