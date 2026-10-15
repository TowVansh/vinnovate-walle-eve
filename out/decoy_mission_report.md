# Mission report: decoy-mission

- Outcome: **EVE FOUND** (ended by: found)
- Steps: 61 | time used: 70/600 | battery left: 23/70 | recharges: 1
- Inferred EVE location: [10, 6] with confidence 100%
- Inferred EVE status: dormant (100%)
- Spoofed/inconsistent readings flagged: 1

## Top hypotheses

1. [10, 6]: 100.0%
2. [0, 0]: 0.0%
3. [1, 0]: 0.0%

## Evidence log

- t=0 initial reading: strength 32.28, bearing 64.6 deg (NE)
- clue at [4, 2]: Beacon log: EVE last pinged from the east half
- strategy change: target [6, 1] -> [1, 1] (P=0.05)
- strategy change: target [1, 1] -> [4, 5] (P=0.25)
- strategy change: target [4, 5] -> [6, 1] (P=0.23)
- strategy change: target [6, 1] -> [7, 2] (P=0.31)
- contact failed at [7, 2], cell ruled out
- strategy change: target [7, 2] -> [6, 1] (P=0.43)
- contact failed at [6, 1], cell ruled out
- SPOOF DETECTED: strong signal but no EVE around [6, 2] after 2 contacts; marked as decoy, belief rebuilt without its readings (new best [4, 5], P=0.01)
- strategy change: target [6, 1] -> [4, 5] (P=0.01)
- strategy change: target [4, 5] -> [10, 5] (P=0.03)
- strategy change: target [10, 5] -> [10, 6] (P=0.05)
- contact made with EVE at [10, 6], status dormant

## Recommended next step

Dock with EVE at [10, 6] and escort her back to base [1, 3] (she is dormant: tow, do not wait for a response)

## Map (post-mission debrief, truth revealed)

```
+--------------------+
|  #        ?        |
| *******# #  #      |
| *##D***   #     #  |
| B*** *#           ?|
| #*   *       # #   |
|# # ****** #        |
| #   #  #*E   #  ## |
|# #      #        # |
|     #   #       #  |
|   ?               #|
|  ##           #   #|
|   #      #    #    |
|#   ?#      #       |
| #  ##              |
|     #    #         |
+--------------------+
```
