(define (domain puzzle8)
  (:requirements :strips :typing)
  
  (:types tile position)
  
  (:predicates
    (at ?tile - tile ?pos - position)
    (blank ?pos - position)
    (adjacent ?pos1 ?pos2 - position)
  )
  
  (:action move
    :parameters (?tile - tile ?from ?to - position)
    :precondition (and
      (at ?tile ?from)
      (blank ?to)
      (adjacent ?from ?to)
    )
    :effect (and
      (not (at ?tile ?from))
      (at ?tile ?to)
      (blank ?from)
      (not (blank ?to))
    )
  )
)
