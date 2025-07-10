# Current UX Pain Points and Desired Outcomes

The initial physician UI exposes only minimal functionality. Internal testing and early clinician feedback highlighted the following pain points:

- **Unclear workflow** – early prototypes required prefixing messages with `test:` to order a lab. The UI now provides an action dropdown and searchable test list.
- **Minimal feedback** – failures while loading a case or backend errors show up as browser alerts or silently in logs.
- **Styling shortcomings** – the static two‑column grid lacks responsive behaviour, accessible color contrast and clear separation between panels.
- **No session status** – users cannot easily determine which account is logged in or log out without refreshing the page.

Clinicians also found it confusing to know when a conversation was finished and asked for a clearer cost summary along with the ability to expand the case vignette while chatting.

## Desired UX Outcomes

- **Clarity** – add inline hints or a help panel describing available actions (question, test, diagnosis) and display the current username with a logout button.
- **Accessibility** – adopt high‑contrast styles and ensure the layout works well on tablets. Form controls should include screen‑reader labels.
- **Visual update** – migrate to a modern component framework (for example React with a UI library) and add collapsible panels so users can reference test results or the case summary without cluttering the chat.

Implementing these improvements will make SDBench more approachable for clinicians evaluating the Gatekeeper approach.
