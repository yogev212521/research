(define (domain hanoi)
  (:requirements :strips :typing)
  
  (:types disc peg)
  
  (:predicates
    (on ?d - disc ?p - peg)
    (clear ?p - peg)
  )
  
  (:action move
    :parameters (?d - disc ?from ?to - peg)
    :precondition (and 
      (on ?d ?from)
      (clear ?to)
    )
    :effect (and
      (not (on ?d ?from))
      (on ?d ?to)
      (clear ?from)
      (not (clear ?to))
    )
  )
)
