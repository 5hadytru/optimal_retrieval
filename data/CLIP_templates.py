iNat2021 = {
    "double": {
        "/": [
            lambda classname, superclass: f"A photo of the {classname.lower()}, a type of {superclass.lower()}.",
            lambda classname, superclass: f"The {classname.lower()}, a type of {superclass.lower()}.",
            lambda classname, superclass: f"A photo of a {classname.lower()}, a type of {superclass.lower()}.",
            lambda classname, superclass: f"A {classname.lower()}, a type of {superclass.lower()}.",
            lambda classname, superclass: f"A photo of a {superclass.lower()}, specifically a {classname.lower()}.",
            lambda classname, superclass: f"The coolest kind of {superclass.lower()}: the {classname.lower()}.",
            lambda classname, superclass: f"The {classname.lower()} is such an interesting {superclass.lower()}.",
            lambda classname, _: f"Today I came across this {classname.lower()}.",
            lambda classname, _: f"A high-resolution photo of a {classname.lower()}.",
            lambda classname, superclass: f"A high-resolution photo of a {classname.lower()}, a type of {superclass.lower()}.",
            lambda classname, superclass: f"My favorite type of {superclass.lower()}: the {classname.lower()}.",
            lambda classname, _: f"A photo of the {classname.lower()}.",
            lambda classname, _: f"Today I came across a {classname.lower()}.",
            lambda classname, _: f"Look what I found: a {classname.lower()}"
        ]
    }
}

sun397 = {
    "single": {
        "": [
            lambda classname: f"A photo of the {classname}.",
            lambda classname: f"A photo of my {classname}.",
            lambda classname: f"A photo of a {classname}.",
            lambda classname: f"A photo of the nearby {classname}.",
            lambda classname: f"A photo of a nearby {classname}.",
            lambda classname: f"A {classname}.",
            lambda classname: f"My {classname}.",
            lambda classname: f"The {classname}"
        ],
    },
    "double": {
        "indoor/": [
            lambda indoor, classname: f"An {indoor} view of a {classname}.",
            lambda indoor, classname: f"An {indoor} photo of a {classname}.",
            lambda indoor, classname: f"An {indoor} view of the {classname}.",
            lambda indoor, classname: f"An {indoor} photo of the {classname}.",
            lambda indoor, classname: f"My favorite place, an {indoor} {classname}.",
            lambda indoor, classname: f"My favorite place, the {indoor} {classname}."
        ],
        "outdoor/": [
            lambda outdoor, classname: f"An {outdoor} view of a {classname}.",
            lambda outdoor, classname: f"An {outdoor} photo of a {classname}.",
            lambda outdoor, classname: f"An {outdoor} view of the {classname}.",
            lambda outdoor, classname: f"An {outdoor} photo of the {classname}.",
            lambda outdoor, classname: f"My favorite place to be, an {outdoor} {classname}.",
            lambda outdoor, classname: f"My favorite place to be, the {outdoor} {classname}"
        ],
        "public/": [
            lambda public, classname: f"A photo of the {public} {classname}.",
            lambda public, classname: f"A photo of a {public} {classname}.",
        ],
        "shop/": [
            lambda _, classname: f"A photo of the {classname}.",
            lambda _, classname: f"A photo of a {classname}.",
            lambda _, classname: f"My favorite {classname}."
        ],
        "interior/": [
            lambda interior, classname: f"An {interior} view of a {classname}.",
            lambda interior, classname: f"A photo of an {interior} view of a {classname}.",
            lambda interior, classname: f"An {interior} view of the {classname}.",
            lambda interior, classname: f"A photo of an {interior} view of the {classname}"
        ],
        "exterior/": [
            lambda exterior, classname: f"An {exterior} view of a {classname}.",
            lambda exterior, classname: f"An {exterior} photo of a {classname}.",
            lambda exterior, classname: f"An {exterior} view of the {classname}.",
            lambda exterior, classname: f"An {exterior} photo of the {classname}"
        ],
        "natural/": [
            lambda natural, classname: f"A {natural} {classname}.",
            lambda natural, classname: f"A {natural} {classname}, a body of water.",
            lambda natural, classname: f"A {natural} {classname}, a good place to relax.",
            lambda natural, classname: f"A photo of a {natural} {classname}.",
            lambda natural, classname: f"A photo of a {natural} {classname}, a body of water.",
            lambda natural, classname: f"A photo of a {natural} {classname}, a good place to relax.",
            lambda natural, classname: f"The {natural} {classname}.",
            lambda natural, classname: f"The {natural} {classname}, a body of water.",
            lambda natural, classname: f"The {natural} {classname}, a good place to relax.",
            lambda natural, classname: f"A photo of the {natural} {classname}.",
            lambda natural, classname: f"A photo of the {natural} {classname}, a body of water.",
            lambda natural, classname: f"A photo of the {natural} {classname}, a good place to relax.",
        ],
        "urban/": [
            lambda urban, classname: f"An {urban} {classname}.",
            lambda urban, classname: f"A photo of an {urban} {classname}.",
            lambda urban, classname: f"The {urban} {classname}.",
            lambda urban, classname: f"A photo of the {urban} {classname}.",
        ],
        "seat/": [
            lambda _seat, classname: f"A {_seat} view of my {classname}.",
            lambda _seat, classname: f"A {_seat} view of the {classname}.",
            lambda _seat, classname: f"A {_seat} view of a {classname}.",
            lambda _, classname: f"My {classname}.",
            lambda _, classname: f"The {classname}.",
            lambda _, classname: f"A {classname}"
        ],
        "office/": [
            lambda office, classname: f"An {office} {classname}.",
            lambda office, classname: f"A photo of an {office} {classname}.",
            lambda office, classname: f"My {office} {classname}.",
            lambda office, classname: f"A photo of my {office} {classname}"
        ],
        "sand/": [
            lambda sand, classname: f"A {sand}y {classname}.",
            lambda sand, classname: f"A photo of a {sand}y {classname}.",
            lambda sand, classname: f"A particularly {sand}y {classname}.",
            lambda sand, classname: f"A {classname}, mostly composed of {sand}"
        ],
        "vegetation/": [
            lambda vegetation, classname: f"A {classname} with lots of {vegetation}"
        ],
        "vehicle/": [
            lambda vehicle, classname: f"A {classname} inside of a {vehicle}"
        ],
        "home/": [
            lambda home, classname: f"A {classname} inside of a {home}.",
            lambda home, classname: f"A {classname} inside someone's {home}"
        ],
        "cultivated/": [
            lambda cultivated, classname: f"A photo of a {cultivated} {classname}.",
            lambda cultivated, classname: f"A photo of the {cultivated} {classname}.",
            lambda cultivated, classname: f"A {cultivated} {classname}.",
            lambda cultivated, classname: f"The {cultivated} {classname}"
        ],
        "wild/": [
            lambda wild, classname: f"A photo of a {wild} {classname}.",
            lambda wild, classname: f"A photo of the {wild} {classname}.",
            lambda wild, classname: f"A {wild} {classname}.",
            lambda wild, classname: f"The {wild} {classname}"
        ],
        "broadleaf/": [
            lambda subclass, classname: f"A photo of a {subclass} {classname}.",
            lambda subclass, classname: f"A photo of the {subclass} {classname}.",
            lambda subclass, classname: f"A {subclass} {classname}.",
            lambda subclass, classname: f"The {subclass} {classname}"
        ],
        "needleleaf/": [
            lambda subclass, classname: f"A photo of a {subclass} {classname}.",
            lambda subclass, classname: f"A photo of the {subclass} {classname}.",
            lambda subclass, classname: f"A {subclass} {classname}.",
            lambda subclass, classname: f"The {subclass} {classname}"
        ],
        "establishment/": [
            lambda subclass, classname: f"A {classname} within an {subclass}.",
            lambda subclass, classname: f"A {classname} inside a local {subclass}"
        ],
        "baseball/": [
            lambda subclass, classname: f"A photo of a {subclass} {classname}.",
            lambda subclass, classname: f"A photo of the {subclass} {classname}.",
            lambda subclass, classname: f"A {subclass} {classname}.",
            lambda subclass, classname: f"The {subclass} {classname}"
        ],
        "football/": [
            lambda subclass, classname: f"A photo of a {subclass} {classname}.",
            lambda subclass, classname: f"A photo of the {subclass} {classname}.",
            lambda subclass, classname: f"A {subclass} {classname}.",
            lambda subclass, classname: f"The {subclass} {classname}"
        ],
        "platform/": [
            lambda subclass, classname: f"A photo of a {classname} {subclass}.",
            lambda subclass, classname: f"A photo of the {classname} {subclass}.",
            lambda subclass, classname: f"A {classname} {subclass}.",
            lambda subclass, classname: f"The {classname} {subclass}"
        ],
        "asia/": [
            lambda subclass, classname: f"A photo of a {classname} found in {subclass}.",
            lambda subclass, classname: f"A photo of the {classname} in {subclass}.",
            lambda subclass, classname: f"A {classname} in {subclass}.",
            lambda subclass, classname: f"The {classname}, found in {subclass}.",
            lambda subclass, classname: f"A photo of an {subclass}n {classname}.",
            lambda subclass, classname: f"A photo of the {subclass}n {classname}.",
            lambda subclass, classname: f"An {subclass}n {classname}.",
            lambda subclass, classname: f"The {subclass}n {classname}"
        ],
        "indoor procenium/": [
            lambda subclass, classname: f"A photo of an {subclass} of a {classname}.",
            lambda subclass, classname: f"A photo of the {subclass} of a {classname}.",
            lambda subclass, classname: f"A {classname}, particularly its {subclass}.",
            lambda subclass, classname: f"The {classname}, particularly its {subclass}"
        ],
        "indoor seats/": [
            lambda subclass, classname: f"A photo of an {subclass} of a {classname}.",
            lambda subclass, classname: f"A photo of the {subclass} of a {classname}.",
            lambda subclass, classname: f"A {classname}, particularly its {subclass}.",
            lambda subclass, classname: f"The {classname}, particularly its {subclass}"
        ],
        "reef/": [
            lambda subclass, classname: f"A photo of an {classname} {subclass}.",
            lambda subclass, classname: f"An {classname} {subclass}, a rich habitat.",
            lambda subclass, classname: f"An {classname} {subclass}.",
            lambda subclass, classname: f"A diver took this photo of an {classname} {subclass}"
        ],
        "fan/": [
            lambda subclass, classname: f"A photo of a {subclass} {classname}.",
            lambda subclass, classname: f"A {subclass} {classname}"
        ],
        "plunge/": [
            lambda subclass, classname: f"A photo of a {subclass} {classname}.",
            lambda subclass, classname: f"A {subclass} {classname}"
        ],
        "block/": [
            lambda subclass, classname: f"A photo of a {subclass} {classname}.",
            lambda subclass, classname: f"A {subclass} {classname}"
        ],
        "storage/": [
            lambda subclass, classname: f"A photo of a {classname} which uses {subclass}.",
            lambda subclass, classname: f"A {classname} using {subclass}"
        ],
        "/": [
            lambda _, classname: f"A photo of a {classname}.",
            lambda _, classname: f"A {classname}.",
            lambda _, classname: f"A photo of the {classname}.",
            lambda _, classname: f"The {classname}.",
            lambda _, classname: f"A photo of my {classname}.",
            lambda _, classname: f"My {classname}.",
            lambda _, classname: f"A view of a {classname}.",
            lambda _, classname: f"A view of the {classname}.",
            lambda _, classname: f"A view of my {classname}"
        ]
    }
}

plant_village = {
    "double": {
        "/": [
            lambda disease, classname: f"Unfortunately, this {classname} leaf has {disease}.",
            lambda disease, classname: f"A {classname} leaf with {disease}.",
            lambda disease, classname: f"A photo of a {classname} leaf with {disease}.",
            lambda disease, classname: f"A zoomed-in photo of a {classname} leaf with {disease}.",
            lambda disease, classname: f"A centered photo of a {classname} leaf with {disease}"
        ]
    },
    "single": {
        "background": [
            lambda classname: f"The {classname}.",
        ],
        "healthy": [
            lambda classname: f"A {classname} leaf.",
            lambda classname: f"A photo of a {classname} leaf.",
            lambda classname: f"A zoomed-in photo of a {classname} leaf.",
            lambda classname: f"A centered photo of a {classname} leaf"
        ],
        "": 0
    }
}

cars = {
    "single": {
        "": [
            lambda c: f'A photo of a {c}.',
            lambda c: f'A photo of a {c}, my favorite car.',
            lambda c: f'A photo of a {c}, a cool car.',
            lambda c: f'A photo of the {c}.',
            lambda c: f'A photo of a used {c}.',
            lambda c: f'A photo of my {c}.',
            lambda c: f'I love my {c}!',
            lambda c: f'A photo of my dirty {c}.',
            lambda c: f'A photo of my clean {c}.',
            lambda c: f'A photo of my old {c}.'
        ]
    }
}

dtd = {
    "single": {
        "": [
            lambda c: f'A photo of a {c} texture.',
            lambda c: f'A photo of a {c} pattern.',
            lambda c: f'A photo of a {c} thing.',
            lambda c: f'A photo of a {c} object.',
            lambda c: f'A photo of the {c} texture.',
            lambda c: f'A photo of the {c} pattern.',
            lambda c: f'A photo of the {c} thing.',
            lambda c: f'A photo of the {c} object.',
            lambda c: f'A {c} texture.',
            lambda c: f'A {c} pattern.',
            lambda c: f'A {c} thing.',
            lambda c: f'A {c} object.',
            lambda c: f'The {c} texture.',
            lambda c: f'The {c} pattern.',
            lambda c: f'The {c} thing.',
            lambda c: f'The {c} object.'
        ]
    }
}

gtsrb = {
    "single": {
        "": [
            lambda c: f'A zoomed in photo of a "{c}" traffic sign.',
            lambda c: f'A centered photo of a "{c}" traffic sign.',
            lambda c: f'A close up photo of a "{c}" traffic sign.'
        ]
    }
}

fgvc = {
    "single": {
        "": [
        lambda c: f'A photo of a {c} aircraft.',
        lambda c: f'A photo of a {c}, a type of aircraft.',
        lambda c: f'A centered photo of a {c} aircraft.',
        lambda c: f'My favorite type of aircraft, a {c}.',
        lambda c: f'The coolest aircraft ever made: the {c}.'
        ]
    }
}

resisc45 = {
    "single": {
        "": [
            lambda c: f'Satellite imagery of {c}.',
            lambda c: f'Aerial imagery of {c}.',
            lambda c: f'Satellite photo of {c}.',
            lambda c: f'Aerial photo of {c}.',
            lambda c: f'Satellite view of {c}.',
            lambda c: f'Aerial view of {c}.',
            lambda c: f'Satellite imagery of a {c}.',
            lambda c: f'Aerial imagery of a {c}.',
            lambda c: f'Satellite photo of a {c}.',
            lambda c: f'Aerial photo of a {c}.',
            lambda c: f'Satellite view of a {c}.',
            lambda c: f'Aerial view of a {c}.',
            lambda c: f'Satellite imagery of the {c}.',
            lambda c: f'Aerial imagery of the {c}.',
            lambda c: f'Satellite photo of the {c}.',
            lambda c: f'Aerial photo of the {c}.',
            lambda c: f'Satellite view of the {c}.',
            lambda c: f'Aerial view of the {c}.'
        ]
    }
}

flowers = {
    "single": {
        "": [
            lambda c: f'A zoomed in photo of a {c} flower.',
            lambda c: f'A centered photo of a {c} flower.',
            lambda c: f'A close up photo of a {c} flower.',
            lambda c: f'A photo of a {c}, a type of flower.',
            lambda c: f'A photo of a {c} flower.',
            lambda c: f'A {c}, a type of flower.',
            lambda c: f'A {c} flower.',
            lambda c: f'My favorite kind of flower: the {c} flower',
            lambda c: f'My favorite type of flower: the {c} flower',
            lambda c: f'Today I came across this {c} flower',
            lambda c: f'Look at this beautiful {c} flower!',
        ]
    }
}

imagenet_1k = {
    "single": {
        "": [
            lambda c: f'a bad photo of a {c}.',
            lambda c: f'a photo of many {c}.',
            lambda c: f'a sculpture of a {c}.',
            lambda c: f'a photo of the hard to see {c}.',
            lambda c: f'a low resolution photo of the {c}.',
            lambda c: f'a rendering of a {c}.',
            lambda c: f'graffiti of a {c}.',
            lambda c: f'a bad photo of the {c}.',
            lambda c: f'a cropped photo of the {c}.',
            lambda c: f'a tattoo of a {c}.',
            lambda c: f'the embroidered {c}.',
            lambda c: f'a photo of a hard to see {c}.',
            lambda c: f'a bright photo of a {c}.',
            lambda c: f'a photo of a clean {c}.',
            lambda c: f'a photo of a dirty {c}.',
            lambda c: f'a dark photo of the {c}.',
            lambda c: f'a drawing of a {c}.',
            lambda c: f'a photo of my {c}.',
            lambda c: f'the plastic {c}.',
            lambda c: f'a photo of the cool {c}.',
            lambda c: f'a close-up photo of a {c}.',
            lambda c: f'a black and white photo of the {c}.',
            lambda c: f'a painting of the {c}.',
            lambda c: f'a painting of a {c}.',
            lambda c: f'a pixelated photo of the {c}.',
            lambda c: f'a sculpture of the {c}.',
            lambda c: f'a bright photo of the {c}.',
            lambda c: f'a cropped photo of a {c}.',
            lambda c: f'a plastic {c}.',
            lambda c: f'a photo of the dirty {c}.',
            lambda c: f'a jpeg corrupted photo of a {c}.',
            lambda c: f'a blurry photo of the {c}.',
            lambda c: f'a photo of the {c}.',
            lambda c: f'a good photo of the {c}.',
            lambda c: f'a rendering of the {c}.',
            lambda c: f'a {c} in a video game.',
            lambda c: f'a photo of one {c}.',
            lambda c: f'a doodle of a {c}.',
            lambda c: f'a close-up photo of the {c}.',
            lambda c: f'a photo of a {c}.',
            lambda c: f'the origami {c}.',
            lambda c: f'the {c} in a video game.',
            lambda c: f'a sketch of a {c}.',
            lambda c: f'a doodle of the {c}.',
            lambda c: f'a origami {c}.',
            lambda c: f'a low resolution photo of a {c}.',
            lambda c: f'the toy {c}.',
            lambda c: f'a rendition of the {c}.',
            lambda c: f'a photo of the clean {c}.',
            lambda c: f'a photo of a large {c}.',
            lambda c: f'a rendition of a {c}.',
            lambda c: f'a photo of a nice {c}.',
            lambda c: f'a photo of a weird {c}.',
            lambda c: f'a blurry photo of a {c}.',
            lambda c: f'a cartoon {c}.',
            lambda c: f'art of a {c}.',
            lambda c: f'a sketch of the {c}.',
            lambda c: f'a embroidered {c}.',
            lambda c: f'a pixelated photo of a {c}.',
            lambda c: f'itap of the {c}.',
            lambda c: f'a jpeg corrupted photo of the {c}.',
            lambda c: f'a good photo of a {c}.',
            lambda c: f'a plushie {c}.',
            lambda c: f'a photo of the nice {c}.',
            lambda c: f'a photo of the small {c}.',
            lambda c: f'a photo of the weird {c}.',
            lambda c: f'the cartoon {c}.',
            lambda c: f'art of the {c}.',
            lambda c: f'a drawing of the {c}.',
            lambda c: f'a photo of the large {c}.',
            lambda c: f'a black and white photo of a {c}.',
            lambda c: f'the plushie {c}.',
            lambda c: f'a dark photo of a {c}.',
            lambda c: f'itap of a {c}.',
            lambda c: f'graffiti of the {c}.',
            lambda c: f'a toy {c}.',
            lambda c: f'itap of my {c}.',
            lambda c: f'a photo of a cool {c}.',
            lambda c: f'a photo of a small {c}.',
            lambda c: f'a tattoo of the {c}.',
        ]
    }
}