(define (problem straight-line)
    (:domain rooms)
    (:objects start-room
              room-1
              room-2
              room-3
              goal-room - room
              bob - person)
    (:init (at bob start-room)
           (connected start-room room-1)
           (connected room-1 room-2)
           (connected room-2 room-3)
           (connected room-3 goal-room))
    (:goal (at bob goal-room)))