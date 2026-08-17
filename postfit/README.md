# PostFit status

This directory preserves the existing downstream projection and plotting
sources, including the user-adjusted presentation styles. Downstream migration
is intentionally outside the current amplitude-fit architecture refactor.

The default `Makefile` therefore builds only `bin/Fit.exe`. The legacy
`PostFit.cu` still expects the removed historical fit-result reader and is not
a supported build target on this branch. Its input migration should be handled
as a separate task against the stable outputs documented in the root README.
