(define (problem gripper-example)
  (:domain gripper)

  (:objects
    room1 room2 - room
    ball1 ball2 - ball
    G1 G2 - gripper
  )

  (:init
    ;; Robby starts in room1
    (at-robby room1)
    ;; Balls are initially in room1
    (at ball1 room1)
    (at ball2 room1)
    ;; All grippers are initially free
    (clear ball1)
    (clear ball2)
    (free G1)
    (free G2)
    (global)
  )

  (:goal
    (and
      ;; Goal: ball1 in room2
      (at ball1 room2)
      ;; Goal: ball2 in room2
      (at ball2 room2)
    )
  )
)
