# X Bulk Delete (Chrome extension)

A local Chrome extension that bulk deletes your own posts and replies on x.com, and can undo your reposts. It clicks through X's own interface for you ("…" menu, then Delete, then Confirm), so it needs no API keys and sends nothing anywhere else.

## Install

1. Download this `x-bulk-delete` folder to your computer.
2. In Chrome, open `chrome://extensions`.
3. Turn on **Developer mode** (top right).
4. Click **Load unpacked** and select the `x-bulk-delete` folder.
5. Pin the extension from the puzzle-piece menu so it's easy to reach.

## Use

1. Log in to x.com.
2. Click the extension icon, then **Go to my Replies**. The Replies tab lists both your posts and your replies. To work through original posts and reposts only, use **Go to my Posts**.
3. Pick what to remove and any filters (date range, keywords).
4. Run a **Dry run** first: it outlines in red what would be deleted and deletes nothing.
5. Untick Dry run and click **Start**. A progress box appears at the bottom left of the page with a **Stop** button. Keep the tab open and visible while it runs.

## Good to know

- **Deletion is permanent.** Download your archive first if you want a copy (Settings → Your account → Download an archive of your data).
- **Pace:** the default delay is 1.5 s between deletes, plus random jitter. Going much faster can trigger X's rate limits or a temporary lock.
- **Old posts:** X's timeline only loads roughly your latest 3,200 posts. If older ones remain, wait a while, reload, and run again. Older posts may stay unreachable from the timeline.
- **Other people's posts** in reply threads are never touched. Only posts whose author handle matches your logged-in account are deleted.
- If X changes its page structure, the extension may stop finding buttons. Errors are counted in the progress box.
