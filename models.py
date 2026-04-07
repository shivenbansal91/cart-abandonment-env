from dataclasses import dataclass

@dataclass
class CartAction:
    action: int  # 0=wait, 1=notify, 2=discount_5, 3=discount_10, 4=discount_20

@dataclass
class CartObservation:
    time_since_abandon: int
    cart_value: int
    discount_given: int
    product_discount: int
    notified: bool    # has agent already sent a notification?
    done: bool
    reward: float