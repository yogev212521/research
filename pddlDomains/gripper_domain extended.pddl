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
  )

  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (not (holding ?x)) (not (clear ?y)) (handempty) (on ?x ?y) (clear ?x)(probabilistic 0.5 (tests)))
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
    )
    :effect (and
      (carry ?b ?g)
      (not (at ?b ?r))
      (not (free ?g))
    )
  )

  (:action drop
    :parameters (?b - block ?r - room ?g - gripper)
    :precondition (and
      (at-robby ?r)
      (carry ?b ?g)
    )
    :effect (and
      (at ?b ?r)
      (free ?g)
      (not (carry ?b ?g))
    )
  )
)
