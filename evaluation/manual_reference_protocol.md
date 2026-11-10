# Manual reference: timing a hand-made evidence brief

**Purpose.** This protocol measures how long the manual research takes without our system. The result fills the **measured pain** in [CHARTER.md](../CHARTER.md) (*Problem and user*), and it is the **manual reference for success criterion 8** (at least 25% median time saved per evidence brief). Results go in [`manual_reference_times.csv`](manual_reference_times.csv).

## Design

| | |
| --- | --- |
| **Task** | one *evidence brief* for one opportunity (neighbourhood × room type × bedrooms), the same task the system is timed on for criterion 8 |
| **Participants** | 2 team members, ideally those who know the data least, so the time is not artificially short |
| **Briefs** | 3 opportunities each, **6 in total**, picked in advance from the opportunity table, e.g. ranks 1, 5, 10, 20, 30 and 40. Each person does different opportunities, so a second brief is not faster because the area was just researched |
| **Allowed tools** | what an investor has: airbnb.com and its filters, Google Maps, the public Inside Airbnb website, a spreadsheet or notes |
| **Not allowed** | anything from the project: the system, the notebooks, the opportunity table |
| **Time limit** | 60 minutes per brief. If it is reached, record `60+` and note how far the person got |
| **Pace** | normal, as an investor would work: no rushing, no polishing |
| **Report** | the median of the 6 times, the number of briefs and participants. A small sample, so the result is **exploratory** |

**For criterion 8 later:** the system (the opportunity table, and the agent) is timed on **the same 6 opportunities**, by people who did not do them manually.

## Before starting the stopwatch (not timed)

1. Write down the opportunity, for example *De Pijp – Rivierenbuurt, entire home, 2 bedrooms*.
2. Open an empty brief sheet (template below).
3. Use the **same search settings for every brief.** Airbnb shows prices only for specific dates. Use, for example, 2 nights starting on a Tuesday next month, 2 guests for a room or 4 for an entire home. Record the settings used in the CSV.

## Step by step (timed)

Start the stopwatch when the browser is opened.

1. **Find the neighbourhood's boundaries.** Look the neighbourhood up on Google Maps or on Inside Airbnb's Amsterdam map, to see roughly which streets it covers.
2. **Search Airbnb.** On airbnb.com, search for *Amsterdam* with the agreed dates and guests. Set the filters:
   - type of place: *Entire home* or *Room*;
   - bedrooms: the opportunity's number (0–1, 2 or 3+).

   Move and zoom the map so that it shows only that neighbourhood.
3. **Count the comparables.** Note how many listings Airbnb shows in that map area ("X places"). This is the size of the opportunity.
4. **Collect 10 comparable listings.** Open 10 listings in the area, one by one. For each, write down:
   - the price per night (the nightly price shown, before fees);
   - the total number of reviews and the rating;
   - the date of the most recent review, as a rough sign of activity.
5. **Work out the typical price.** Compute the **median** of the 10 nightly prices, and note the cheapest and the most expensive.
6. **Judge how busy they are.** Of the 10 listings, count how many have a review in the last 3 months, and note the typical number of reviews. Optional: open the calendars of 3 listings and count the blocked (booked) days in the next 30 days.
7. **Count the competition.** Remove the type and bedroom filters, keeping the dates and the map area, and note how many listings of any kind the area shows.


Stop the stopwatch.





## Filling the charter

- **Median minutes per brief:** the middle value of the 6 recorded times. For an even number of briefs, take the average of the two middle values.
- **Number of opportunities:** the number of briefs timed (6).
- **Hours for a city's top 10:** median minutes × 10 ÷ 60. For example, a median of 25 minutes gives about 4 hours.
- Add the sample: *"6 briefs by 2 team members; a small sample, so exploratory."*
- Put the same median in criterion 8 as the manual reference.
