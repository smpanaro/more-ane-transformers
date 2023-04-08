#!/usr/bin/env bash

# log stream --predicate '(process IN { "aned", "ANECompilerService", "Python" }
log show --last 10m --debug --predicate '(process IN { "aned", "ANECompilerService", "Python" }
    && NOT (eventMessage CONTAINS "libORTools.dylib")
    && NOT (eventMessage CONTAINS "Underflow detected")
    && NOT (eventMessage CONTAINS "Invalid Interleave setting"))
 OR (sender in { "ANEServices" } && NOT (process IN { "mediaanalysisd" }))
 OR (process in { "kernel" }
    && NOT (eventMessage CONTAINS "Sandbox")
    && NOT (eventMessage CONTAINS "IOPlatformSleepAction")
    && ((eventMessage CONTAINS "ane0") OR (eventMessage CONTAINS "ANE0") OR (eventMessage CONTAINS "DART"))
 )
 OR (eventMessage CONTAINS "espresso")'