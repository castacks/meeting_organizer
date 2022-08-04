"""
Script that organizes meetings.

Provide this script:
    1. A csv of the current AirLab roster
    2. A yaml file containing the format of the meeting
    3. A yaml file containing the list of meeting dates

This script will then output a meeting schedule where the following will be obeyed:
    1. No student will present 2+ times more than any other student

SAM TODO: think about how to do constraints
"""

import os
import argparse
import yaml
import datetime
import numpy as np
import pandas as pd

def generate_schedule(
        roster,
        prev_meetings,
        meeting_format,
        meeting_schedule
    ):
    """
    Generate a meeting schedule
    Args:
        roster: csv containing the list of people to give talks
        prev_meetings: df containing previous meetings
        meeting_format: dict of meeting formats
        meeting_scedule: list of meeting dates

    Returns:
        schedule: df of schedule that obeys the following constraints:
            1. No student presents 2+ times more than any other student in a year
            2. No student presents twice in a day
            3. Attempt to space presentations out as much as possible for a given student
    """
    #set up the df
    cols = ["Date", ""]
    for k in meeting_format:
        cols.extend([k] + meeting_format[k] + [""])
    res = {}

    #get the number of times each student presented for each group in the last year
    presentation_counts = {k:{k2:0 for k2 in roster['Name']} for k in meeting_format.keys()}

    #get the last time each student presented in each group
    presentation_recency = {k:{k2:100000 for k2 in roster['Name']} for k in meeting_format.keys()}

    for df in prev_meetings:
        for i, row in df.iterrows():
            rowdate = datetime.datetime.strptime(row["Date"], "%Y-%m-%d").date()
            first_new_date = min(meeting_schedule)
            #process the row entry if it is within one year of the first new meeting time
            if (rowdate < first_new_date) and ((first_new_date - rowdate).days < 365):
                for k in meeting_format.keys():
                    if k in row.keys():
                        kidx = row.keys().to_list().index(k) + 1
                        while not ((str(row[kidx]) == 'nan') or (kidx >= len(row))):
                            person = row[kidx]
                            if person in presentation_counts[k].keys():
                                presentation_counts[k][person] += 1
                                presentation_recency[k][person] = min(presentation_recency[k][person], (first_new_date - rowdate).days)
                            kidx += 1

    #for each group, first sort students by recency, then count to assign
    for group in meeting_format.keys():
        res[group] = {}
        people = list(roster['Name'])
        np.random.shuffle(people)

        #create a priority score in each group (lower = higher priority)
        cnts = np.array([presentation_counts[group][k] for k in people])
        recency = np.array([presentation_recency[group][k] for k in people])
        recency = recency.max() - recency
        scores = recency * cnts.max() + cnts
        idxs = np.argsort(scores)

        #put people in groups by iterating through priority scores
        cnt = 0
        for date in meeting_schedule:
            res[group][date] = {}
            for i, k in enumerate(meeting_format[group]):
                res[group][date][k] = people[idxs[cnt % len(people)]]
                cnt += 1

    #build the dataframe
    out_df = []
    for date in meeting_schedule:
        rows = [""] * len(cols)
        rows[0] = date
        for k in res.keys():
            res2 = res[k][date]
            for i, (event, speaker) in enumerate(res2.items()):
                bidx = cols.index(k)
                rows[bidx + i + 1] = speaker

        out_df.append(rows)

    out_df = pd.DataFrame(out_df, columns=cols)
    return out_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meetings_dir', type=str, required=True, help='path to directory containing past meetings')
    parser.add_argument('--roster_fp', type=str, required=True, help='path to the csv containing the current roster')
    parser.add_argument('--format_fp', type=str, required=True, help='path to the yaml containing the meeting format')
    parser.add_argument('--schedule_fp', type=str, required=True, help='path to the yaml containing the meeting schedule')
    args = parser.parse_args()

    meeting_fps = os.listdir(args.meetings_dir)
    #dont concat as meeting formats can change
    prev_meetings = [pd.read_csv(os.path.join(args.meetings_dir, fp)) for fp in meeting_fps]

    roster = pd.read_csv(args.roster_fp)
    meeting_format = yaml.safe_load(open(args.format_fp, 'r'))
    meeting_schedule = yaml.safe_load(open(args.schedule_fp, 'r'))

    new_schedule = generate_schedule(roster, prev_meetings, meeting_format, meeting_schedule)
    start_date = new_schedule["Date"].iloc[0]
    end_date = new_schedule["Date"].iloc[-1]
    fp = '{}_to_{}.csv'.format(start_date, end_date)

    print('OLD MEETINGS:')
    for pdf in prev_meetings:
        print(pdf)

    print('NEW SCHEDULE:')
    print(new_schedule)

    inp = input('Save schedule? [Y/n]')
    if inp == 'n':
        exit(0)

    new_schedule.to_csv(os.path.join(args.meetings_dir, fp), index=False)
