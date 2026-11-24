# Labelling guide: does this table meet this need?

You are labelling pairs of (need, table) in `hand_labels.csv`. Fill the last column, `your_grade`,
with 0, 1, 2 or 3. The sheet does not show what the automatic judge said, on purpose. Do not look
it up.

## The four grades

| Grade | Meaning | Test to apply |
|---|---|---|
| 3 | Meets the need | The robot, the task and every stated requirement (size, cameras, frame rate, format) match what the table says. You would send this address to a colleague without a caveat. |
| 2 | Usable | The task or the robot matches and nothing the table says contradicts the need, but one stated requirement is not confirmed by the table's text. You would send it with "check the X first". |
| 1 | Related only | Same robot or the same kind of task, but not what was asked for; or a stated requirement is contradicted (asked for 200 episodes, the table has 10). |
| 0 | Not relevant | Something else, or the table says too little to tell what it holds. |

## Hard cases, decided once

1. **The table does not say what task it shows.** It cannot get more than 1 for a need about a
   task, even when the robot is right. Most tables are like this. "so101, 50 episodes, 30 fps" is
   not evidence of pick-and-place.
2. **The need states a number.** "At least 200 episodes" with a table of 180 is a contradiction:
   grade 1, not 2.
3. **The need names no robot.** Any robot is fine; grade on the task and the other requirements.
4. **A benchmark by name** (LIBERO, push-T, DROID). A conversion of that benchmark is a 3. A table
   "inspired by" it or a different benchmark of the same kind is a 1.
5. **Scratch names** ("test", "tmp") do not lower the grade by themselves. Grade what the table
   says it holds. If it says nothing, that is rule 1.
6. **A need with no answer** (data this catalog cannot hold). Grade the table that was returned on
   its merits: an image-mask dataset is a 0 for "MRI brain scans with segmentation masks".
7. **You opened the address to decide.** Allowed, and note it in a comment column you may add.
   The automatic judge could not; these pairs show what the text alone hides.

## How to work

- Label in two sittings of about 100. Stop when you notice you are skimming.
- Use the table text in the sheet first. Open the address only when the text is not enough.
- If you change your mind about a rule half way, write the new rule down here and re-label the
  pairs it affects. Do not let the rule drift silently.
- When done, run `python scripts/vxfind_eval.py judge`. It reads the sheet, never overwrites a
  sheet that has grades in it, and reports agreement between you and the judge.
