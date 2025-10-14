(define (domain blocksworld)
  (:requirements
  :strips
  :typing
  :disjunctive-preconditions
  :negative-preconditions)

  (:types block)
  (:predicates
    (on ?x - block ?y - block)
    (ontable ?x - block)
    (clear ?x - block)
    (handempty)
    (tests)

    (holding ?x - block)
  )

  (:action pick-up
    :parameters (?x - block)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x)) (not (clear ?x))(not (handempty)) (holding ?x))
  )

  (:action put-down
    :parameters (?x - block)
    :precondition (and (holding ?x) (tests))
    :effect (and (ontable ?x) (clear ?x) (handempty) (not (holding ?x)) (probabilistic 0.5 (not (tests))))
  )

  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y) (not (tests)))
    :effect (and (not (holding ?x)) (not (clear ?y)) (handempty) (on ?x ?y) (clear ?x)(probabilistic 0.5 (tests)))
  )

  (:action unstack
    :parameters (?x - block ?y - block)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x) (clear ?y) (not (on ?x ?y)) (not (clear ?x)) (not (handempty)))
  )
)
