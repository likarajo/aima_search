Rajarshi Chattopadhyay - rxc170010@utdallas.edu
AI CS6364- Homework 1

Files:
1) ms_ca.py - For Q2.
2) road_trip.py - For Q3.
 a) _roadsgraph.txt - road graph (in miles)
 b) _heuristics.txt - direct/flight distances to dallas (in miles)

Programming language - Python3

Operating System - Unix

Aima code reference: https://github.com/aimacode/aima-python/blob/master/
Main file used: search.py, which internally uses utils.py (to be kept in same directory)

Codes:
1) ms_ca.py - Missionaries and Cannibals Problem

Performs the following operations
 (a) uniform-cost search
 (b) iterative deepening search
 (c) greedy best-first search
 (d) A* search
 (e) recursive best-first search

Input format:
`python3 ms_ca.py missionaryCount cannibalsCount > outputFile`

All parameters are mandatory.
It is recommended to redirect the output to a file as the search space can be large.

Example:
`python3 ms_ca.py 3 3 > ms_ca_out.txt`

Operation details:
- This program prints the path to goal
- It expands and shows the lists and frontier for all the visited nodes for A*, greedy best-first search and uniform-cost search.
- It prints f_limit, best, alternative, current-city and next-city for each node visited for recursive best-first search.
- Each state between path to goal will be in the form of a collection of 5 attributes:-
    - missionary count on the initial side
    - cannibal count on the initial side
    - missionary count on the goal side
    - cannibal count on the goal side
    - current boat location
- Boat location could be either L or initial side, or R or the goal side.
- During state transition the boat has to move to the other side.
- Cannibal count at any side can never be more than the missionary count.

2) road_trip.py: Search for road trips

Performs the following operations
 (a) A* search
 (b) Recursive best-first search

Input format:
`python3 road_trip.py heuristicInputFile roadGraphFile "sourceCity" "destinationCity" > outputFile`

All parameters are mandatory.
It is recommended to redirect the output to a file as the search space can be large.

Example:
`python3 road_trip.py "_heuristics.txt" "_roadsgraph.txt" "Seattle" "Dallas" > road_trip_out.txt`

Operation details:
- This program first checks if the given heuristic is consistent.
- Then it performs A* search and recursive best-first search on the given graph to find out a path between source city to destination city.
- For A* search it prints current node, evaluation function of current node, frontier and the expanded sets for each visited nodes.
- For Recursive best-first search it prints f_limit, best, alternative, current-city and next-city for each node visited.
