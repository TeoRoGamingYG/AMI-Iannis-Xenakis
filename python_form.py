import numpy as np
import random
import pretty_midi
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1.  Datele de intrare ---

DATA_INPUT = (
"000000011300022600033900045100056400067600078900090100101300112500123600"
"134800145900156900168000179000190000200900211800222700233500244300255000"
"265700276300286900297400307900318300328600338900349100359300369400379400"
"389300399200409000418700428400438000447500456900466200475500484700493700"
"502700511700520500529200537900546500554900563300571600579800587900595900"
"603900611700619400627000634600642000649400656600663800670800677800684700"
"691400698100704700711200717500723800730000736100742100748000753800759500"
"765100770700776100781400786700791800796900801900806800811600816300820900"
"825400829900834200838500842700846800850800854800858600862400866100869800"
"873300876800880200883500886800890000893100896100899100902000904800907600"
"910300913000915500918100920500922900925200927500929700931900934000936100"
"938100940000941900943800945700947300949000950700952300953800955400956900"
"958300959700961100962400963700964900966100967300968400969500970600971600"
"972600973600974500975500976300977200978000978800979600980400981100981800"
"982500983200983800984400985000985600986100986700987200987700988200988600"
"989100989500989900990300990700991100991500991800992200992500992800993100"
"993400993700993900994200994400994700994900995100995300995500995700995900"
"996100996300996400996600996700996900997000997200997300997400997500997600"
"997700997900997960998050998140998230998320998400998480998550998620998680"
"998740998800998850998910998960999010999060999100999140999180999230999270"
"999300999340999370999400999440999470999500999530999560999580999600999630"
"999650999670999690999700"
"255099970000263099980000275099990000313099999000346099999900377099999990"
"406099999999100E30100000000"
"04005010020017730035563177245390100000071000000000112000"
"0000150500500120720001600025000120101020309020201010202"
"000000000"
"01010000100700101000010090010100001012001010000101100101000010090"
"01010000101200101000010080010100001008001010000101200101000010080"
"01010000150200101000020020"
"1755000010999"
"3975000015999"
"29710000206001754000010400"
"348500001540015630000154001953000010200"
"39750000151502971000010090175400000709017550000100903363000010090"
"1953000010070101300001020034850000152001563000015020"
"00003467005000000154800500"
"00003467005000000154800500"
"0000326810999"
"0000336310999"
"00001953108000000101307200"
"00003487155000000157215500"
"25080408011309"
"08071602010110"
"03030420010110"
"02050325010112"
"03350315011505"
"02100302103907"
"02020203150207"
"02020202410207"
"03090317041609"
"03132003200509"
"02052801030409"
"45011202020106"
)
GLOBAL_POSITION = 0

TIMBRE_PROGRAMS = {
    0: [  # 1. Percussion
        117,  # Taiko Drum
        115,  # Woodblock
        118   # Melodic Tom
    ],
    1: [60],        # 2. French horn
    2: [73],        # 3. Flute
    3: [71, 72],    # 4. Clarinet / Bass Clarinet
    4: [40, 42],    # 5. Glissando (Violin / Cello)
    5: [73, 71, 60, 56, 57],  # 6. Tremolo / flutter (winds & brass)
    6: [45, 46],    # 7. Pizzicato strings
    7: [40, 42],    # 8. Struck strings (approx)
    8: [11],        # 9. Vibraphone
    9: [56],        # 10. Trumpet
    10: [57],       # 11. Trombone
    11: [40, 42]    # 12. Bowed strings
}

# --- 2. Citirea input-ului ---

def read_fixed_width(width, decimal_places=0, is_integer=False):
    global DATA_INPUT, GLOBAL_POSITION

    start = GLOBAL_POSITION
    end = start + width

    field_data = DATA_INPUT[start:end]
    GLOBAL_POSITION = end

    cleaned_data = field_data.strip()

    if is_integer:
        return int(cleaned_data)

    try:
        if 'E' in cleaned_data.upper() or '.' in cleaned_data:
            return float(cleaned_data)

        if decimal_places == 0:
            return float(cleaned_data)

        if len(cleaned_data) < decimal_places:
            cleaned_data = '0' * (decimal_places - len(cleaned_data)) + cleaned_data

        insert_point = len(cleaned_data) - decimal_places
        value_str = cleaned_data[:insert_point] + '.' + cleaned_data[insert_point:]
        return float(value_str)

    except ValueError:
        # ex: 'E30100'
        raise ValueError(f"Eroare de Conversie/Aliniere: '{field_data}' (W={width}, D={decimal_places})")


def read_array_of_format(count, width, decimals, is_integer=False):
    return [read_fixed_width(width, decimals, is_integer) for _ in range(count)]


# Pas 1: Tabela TETA (F6.6, 256 valori)
TETA = read_array_of_format(256, 6, 6)

# Pas 2: Tabele Z1/Z2 (F3.2, F9.8, E6.2, F9.8)
Z1 = []
Z2 = []
for _ in range(7):
    Z1.append(read_fixed_width(3, 2))
    Z2.append(read_fixed_width(9, 8))

# Z1(8) (E6.2) si Z2(8) (F9.8)
Z1_8_RAW = DATA_INPUT[GLOBAL_POSITION: GLOBAL_POSITION + 6]
GLOBAL_POSITION += 6
Z1.append(float(Z1_8_RAW))
Z2.append(read_fixed_width(9, 8))

# Pas 3: Constante globale
DELTA = read_fixed_width(3, 0)
V3 = read_fixed_width(3, 3)

# Citirea constantelor A (5 * F3.1)
A10 = read_fixed_width(3, 1)
A20 = read_fixed_width(3, 1)
A17 = read_fixed_width(3, 1)
A30 = read_fixed_width(3, 1)
A35 = read_fixed_width(3, 1)

BF = read_fixed_width(2, 0)
SQPI = read_fixed_width(8, 7)
EPSI = read_fixed_width(8, 8)
VITLIM = read_fixed_width(4, 2)
ALEA = read_fixed_width(8, 8)
ALIM = read_fixed_width(5, 2)

# Pas 4: Parametri de control (5I3, 2I2, 2F6.0, 12I2)
KT1, KT2, KW, KNL, KTR = read_array_of_format(5, 3, 0, is_integer=True)
KTE, KR1 = read_array_of_format(2, 2, 0, is_integer=True)
GTNA, GTNS = read_array_of_format(2, 6, 0)
NT_VECTOR = read_array_of_format(12, 2, 0, is_integer=True)

# Pas 4.1: Parametri KTEST (3I3)
KTEST = read_array_of_format(3, 3, 0, is_integer=True)

# Pas 5: Matricea PN/HAMIN

for i in range(KTR):
    KTS = NT_VECTOR[i]
    if KTS > 0:
        for j in range(KTS):
            read_array_of_format(5, 2, 0) # 5 * F2.0 (Pitch/GN)
            read_fixed_width(3, 3) # F3.3 (PN)

# Pas 6: Matricea E(I, J) (12 clase * 7 nivele * F2.2)

E_Matrix = read_array_of_format(KTR * KTE, 2, 2)
E_Matrix_Reshaped= np.array(E_Matrix).reshape(KTR, KTE)

# --- 3. Generarea Scriptului ---

class Event:
    def __init__(self, start, duration, pitch, velocity, instrument):
        self.start = start
        self.duration = duration
        self.pitch = pitch
        self.velocity = velocity
        self.instrument = instrument

class FreeStochasticMusic:
    def __init__(self, TETA, Z1, Z2, E, KTR, KTE, V3, DELTA, VITLIM):
        self.TETA = TETA
        self.Z1 = Z1
        self.Z2 = Z2
        self.E = E
        self.KTR = KTR
        self.KTE = KTE
        self.V3 = V3
        self.DELTA = DELTA
        self.VITLIM = VITLIM

        self.time = 0.0
        self.events = []
        self.time_window = 2.0

    # Densitate (Poisson)
    def choose_density_level(self):
        return random.randint(0, self.KTE - 1)

    # Clasa timbru (matricea E)
    def choose_timbre_class(self, density_level):
        probs = self.E[:, density_level]
        probs = probs / np.sum(probs)
        return np.random.choice(self.KTR, p=probs)

    # Durata exponentiala
    def choose_duration_class(self):
        return random.randint(0, len(self.Z1) - 2)

    def generate_duration(self):
        k = self.choose_duration_class()
        raw = np.random.exponential(scale=self.Z1[k]) * self.Z2[k]
        duration = min(raw, self.VITLIM)
        duration = max(duration, self.DELTA * 0.01)
        return duration

    # PITCH (din TETA)
    def generate_pitch(self, density):
        centers = [48, 55, 62, 67, 72]  # registre orchestrale
        center = random.choice(centers)
        
        delta_index = np.random.choice(len(self.TETA))
        delta = (self.TETA[delta_index] - 0.5) * 8

        spread = 1 + density  # densitate mare = cluster mai larg
        pitch = int(center + delta + np.random.normal(0, spread))

        return np.clip(pitch, 20, 108)

    # Intensitate
    def generate_velocity(self, density):
        base = 40 + density * 12
        velocity = base + random.randint(0, 5)
        return int(np.clip(velocity, 30, 120))

    def generate_event(self):
        density = self.choose_density_level()
        timbre = self.choose_timbre_class(density)

        duration = self.generate_duration()
        pitch = self.generate_pitch(density)
        velocity = self.generate_velocity(density)
        
        X = np.random.rand()
        max_interval = 0.5
        inter_time = min(-np.log(X) / (self.V3 * np.exp(density)), max_interval)  # DA = V3*exp(U)
        start = self.time + inter_time
        self.time = start

        ev = Event(
            start=start,
            duration=duration,
            pitch=pitch,
            velocity=velocity,
            instrument=timbre
        )

        self.events.append(ev)

    def generate(self, total_time=60):
        while self.time < total_time:
            density = self.choose_density_level()

            # rata Poisson (mapare)
            lam = (density + 1) * 2.0

            num_events = np.random.poisson(lam)

            for _ in range(num_events):
                self.generate_event()


        return self.events

# export MIDI

def events_to_midi(events, filename="xenakis_free_stochastic_8.mid"):
    midi = pretty_midi.PrettyMIDI()
    instruments = {}

    for ev in events:
        timbre_class = ev.instrument

        # alegem un program MIDI din clasa
        program = random.choice(TIMBRE_PROGRAMS[timbre_class])

        key = (timbre_class, program)

        if key not in instruments:
            instruments[key] = pretty_midi.Instrument(program=program)

        note = pretty_midi.Note(
            velocity=ev.velocity,
            pitch=int(ev.pitch),
            start=float(ev.start),
            end=float(ev.start + ev.duration)
        )

        instruments[key].notes.append(note)

    for instr in instruments.values():
        midi.instruments.append(instr)

    midi.write(filename)
    print(f"MIDI Atrees-like: {filename}")

# Rulare
fsm = FreeStochasticMusic(
    TETA=TETA,
    Z1=Z1,
    Z2=Z2,
    E=E_Matrix_Reshaped,
    KTR=KTR,
    KTE=KTE,
    V3=V3,
    DELTA=DELTA,
    VITLIM=VITLIM
)

events = fsm.generate(total_time=90)
events_to_midi(events)

print("Numar evenimente:", len(events))
print("Primul eveniment:", vars(events[0]))
print("Ultimul eveniment:", vars(events[-1]))