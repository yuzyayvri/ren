from scripts.phase3_ap50 import ap50


def test_confidence_order_and_one_to_one_matching():
 r=[('a',[(0,(0,0,10,10))],[(0,(0,0,10,10),.2),(0,(0,0,10,10),.9)])]
 per,macro=ap50(r); assert per[0]>0 and macro==sum(per)/3
def test_empty_and_duplicate_classes_are_deterministic():
 assert ap50([('a',[],[])])[0][0]==0.0
 assert ap50([('a',[(0,(0,0,1,1))],[])])[0][0]==0.0
def test_macro_averaging():
 per,macro=ap50([('a',[(0,(0,0,1,1))],[(0,(0,0,1,1),.9)]),('b',[(1,(0,0,1,1))],[(1,(0,0,1,1),.9)])])
 assert macro==sum(per)/3
