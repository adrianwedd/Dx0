# Current UX Pain Points and Desired Outcomes

The initial physician UI exposes only minimal functionality. Internal testing and early clinician feedback highlighted the following pain points:

- **Unclear workflow** – ordering a test requires prefixing messages with `test:` yet the interface offers no hint about this syntax.
- **Minimal feedback** – failures while loading a case or backend errors show up as browser alerts or silently in logs.
- **Styling shortcomings** – the static two‑column grid lacks responsive behaviour, accessible color contrast and clear separation between panels.
- **No session status** – users cannot easily determine which account is logged in or log out without refreshing the page.

- **Conversation status** – clinicians were unsure when chats were complete or how to view the final cost.
- **Expandable vignette** – they wanted to expand the case description while chatting.

## Desired UX Outcomes

- **Clarity** – add inline hints or a help panel describing available commands (`test:` and `diagnosis:`) and display the current username with a logout button.
- **Accessibility** – adopt high‑contrast styles and ensure the layout works well on tablets. Form controls should include screen‑reader labels.
- **Visual update** – migrate to a modern component framework (for example React with a UI library) and add collapsible panels so users can reference test results or the case summary without cluttering the chat.

Implementing these improvements will make SDBench more approachable for clinicians evaluating the Gatekeeper approach.
