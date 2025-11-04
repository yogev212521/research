(define (domain gripper)
  (:requirements :strips :typing :negative-preconditions)

  (:types
    room ball gripper
  )

  (:predicates
    (at-robby ?r - room)
    (at ?b - ball ?r - room)
    (free ?g - gripper)
    (carry ?b - ball ?g - gripper)
    (global)
  )

  (:action move
    :parameters (?from - room ?to - room)
    :precondition (at-robby ?from)
    :effect (and
      (not (at-robby ?from))
      (at-robby ?to)
    )
  )

  (:action pick
    :parameters (?b - ball ?r - room ?g - gripper)
    :precondition (and
      (at-robby ?r)
      (at ?b ?r)
      (free ?g)
      (not (carry ?b ?g))
      (global)
    )
    :effect (and
      (carry ?b ?g)
      (not (at ?b ?r))
      (not (free ?g))
      (not (global))
    )
  )

  (:action drop
    :parameters (?b - ball ?r - room ?g - gripper)
    :precondition (and
      (at-robby ?r)
      (carry ?b ?g)
      (not (global))
    )
    :effect (and
      (at ?b ?r)
      (free ?g)
      (not (carry ?b ?g))
      (global)
    )
  )
)
