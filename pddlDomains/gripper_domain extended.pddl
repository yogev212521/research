(define (domain gripper)
  (:requirements :strips :typing :negative-preconditions)

  (:types
    room block gripper
  )

  (:predicates
    (at-robby ?r - room)
    (at ?b - block ?r - room)
    (free ?g - gripper)
    (carry ?b - block ?g - gripper)
    (on ?block ?block)
    (clear ?block)
    (global)
  )

  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (not (holding ?x)) (not (clear ?y))(on ?x ?y) (clear ?x))
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
    :parameters (?b - block ?r - room ?g - gripper)
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
    :parameters (?b - block ?r - room ?g - gripper)
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
