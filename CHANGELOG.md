# CHANGELOG

## v1.1.0

- Added data-driven configuration lifecycle through `kc_ipa_options_t`.
- Added `kc_ipa_options_default()`, `kc_ipa_options_load_env()`, and `kc_ipa_options_free()` to the public API.
- Refactored `kc_ipa_open()` to take `kc_ipa_options_t`.
- CLI is now decoupled from `libipa`; configuration is initialized through options, then overridden by flags.
- Added signal listener lifecycle: `kc_ipa_on_signal()`, `kc_ipa_raise_signal()`, `kc_ipa_listen_signals()`, `kc_ipa_listen_signal()`, and `kc_ipa_signal_listener()`.

## v1.0.0

- Published the stable baseline release.
- Provided instruction-coded processor architecture.
- Supported native instruction embedding and execution engine.

---

**Author:** KaisarCode

**Email:** <kaisar@kaisarcode.com>

**Website:** [https://kaisarcode.com](https://kaisarcode.com)

**License:** [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html)

© 2026 KaisarCode
