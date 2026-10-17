"""Labels reflect what the rule-based tracker actually measures."""
EVENT_LABELS = {'cup_moved':'CUP MOVING','cup_returned':'NEAR START POSITION'}
EVENT_DESCRIPTIONS = {
 'cup_moved':'The tracked cup changed position during this interval. This does not establish whether it was lifted or placed.',
 'cup_returned':'The tracked cup moved near its initial position. Contact with the table is not confirmed.'}
VIDEO_DETAILS = {'cup_moved':'Observed cup movement','cup_returned':'Movement near starting position'}
def display_label(label): return EVENT_LABELS.get(label,label.replace('_',' ').upper())
def describe_event(label): return EVENT_DESCRIPTIONS.get(label,'Tracked cup position changed.')
