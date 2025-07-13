# UI Quality Assurance Checklist

This document outlines the process for validating the web interface across supported environments.

## Supported Platforms

- **Browsers:** Chrome, Firefox, Safari, and Edge (latest two releases)
- **Operating Systems:** Windows 11, macOS 14, Ubuntu 22.04
- **Devices:** Desktop, tablet, and mobile viewport sizes

## Testing Checklist

1. **Visual Integrity**
   - Launch the app in each supported browser.
   - Verify that all pages render without layout shifts or missing assets.
   - Compare against design mocks for spacing, colors, and typography.
2. **Functional Correctness**
   - Validate login and logout flows.
   - Exercise chat interactions: asking questions, ordering tests, and entering a diagnosis.
   - Confirm that error states and loading indicators appear when expected.
3. **Responsive Behaviour**
   - Use the browser dev tools to simulate widths from 320 px up to 1920 px.
   - Ensure content remains readable and controls remain usable across breakpoints.
4. **Accessibility**
 - Run `npx axe` or a similar tool on the main pages to detect WCAG issues.
  - Navigate key workflows using only the keyboard and confirm focus styling is visible.
  - Verify that form controls have descriptive labels for screen readers.
  - Execute the automated Playwright suite with `pytest tests/test_e2e_ui_playwright.py` for cross-browser checks.
5. **Defect Tracking**
   - Record any issues in the issue tracker with screenshots and reproduction steps.
   - Prioritize accessibility and functional defects ahead of visual polish.

After fixes land, repeat the relevant tests to ensure no regressions have been introduced.
