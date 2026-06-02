"""RDKit 2D molecular descriptors — organised by category.

Usage::

    from configs.rdkit_descriptors import RECOMMENDED, ALL_2D

    # RECOMMENDED → list of descriptor names (26 commonly used in QSAR)
    # ALL_2D       → dict {category: [(name, description), ...]}

Reference
---------
All descriptors are computed via ``rdkit.Chem.Descriptors`` and do **not**
require 3-D coordinates — a molecular graph parsed from SMILES or InChI is
sufficient.
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Recommended set
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# These 26 descriptors are widely used in QSAR / molecular property prediction
# literature and provide a balanced coverage of molecular properties.

RECOMMENDED: list[str] = [
    # ---- size & weight ----
    "HeavyAtomMolWt",         # molecular weight (heavy atoms only)
    "HeavyAtomCount",          # number of heavy (non-H) atoms
    # ---- lipophilicity ----
    "MolLogP",                 # Wildman-Crippen octanol-water partition coefficient
    "MolMR",                   # Wildman-Crippen molar refractivity
    # ---- polarity / charge ----
    "MaxPartialCharge",        # most positive Gasteiger partial charge
    "MinPartialCharge",        # most negative Gasteiger partial charge
    "MaxAbsPartialCharge",     # max absolute Gasteiger partial charge
    # ---- H-bond ----
    "NumHAcceptors",           # number of hydrogen-bond acceptors
    "NumHDonors",              # number of hydrogen-bond donors
    # ---- topology ----
    "FractionCSP3",            # fraction of sp³ carbons (saturation index)
    "Chi0v",                   # valence molecular connectivity index χ₀ᵛ
    "Chi1v",                   # valence molecular connectivity index χ₁ᵛ
    "Kappa1",                  # Kier shape index ¹κ (cyclicity / branching)
    "Kappa2",                  # Kier shape index ²κ (spatial density)
    "Kappa3",                  # Kier shape index ³κ (central volume)
    "HallKierAlpha",           # Hall-Kier alpha (sum of atomic contributions / atom count)
    "BalabanJ",                # Balaban's J — average distance-sum connectivity
    "BertzCT",                 # Bertz complexity index
    # ---- rings ----
    "RingCount",               # total number of rings
    "NumAromaticRings",        # number of aromatic rings
    "NumSaturatedRings",       # number of saturated rings
    "NumAliphaticRings",       # number of aliphatic rings
    # ---- surface area ----
    "TPSA",                    # topological polar surface area
    "LabuteASA",               # Labute's approximate surface area
    # ---- flexibility ----
    "NumRotatableBonds",       # number of rotatable bonds
    # ---- heteroatoms ----
    "NumHeteroatoms",          # number of heteroatoms (non-C, non-H)
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full catalogue  (217 descriptors)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALL_2D: dict[str, list[tuple[str, str]]] = {
    # ── Molecular Weight & Size ───────────────────────────────────────────
    "Molecular Weight & Size": [
        ("MolWt",              "average molecular weight (including hydrogens)"),
        ("ExactMolWt",         "exact (monoisotopic) molecular weight"),
        ("HeavyAtomMolWt",     "average molecular weight (heavy atoms only, ignores H)"),
        ("HeavyAtomCount",     "number of heavy (non-hydrogen) atoms"),
        ("NumValenceElectrons","total number of valence electrons"),
    ],

    # ── Lipophilicity ──────────────────────────────────────────────────────
    "Lipophilicity": [
        ("MolLogP",            "Wildman-Crippen LogP (octanol-water partition coefficient)"),
        ("MolMR",              "Wildman-Crippen molar refractivity"),
    ],

    # ── Hydrogen Bonding ───────────────────────────────────────────────────
    "Hydrogen Bonding": [
        ("NumHAcceptors",      "number of hydrogen-bond acceptor atoms (N + O + F)"),
        ("NumHDonors",         "number of hydrogen-bond donor atoms (NH + OH)"),
        ("NHOHCount",          "number of NH + OH groups"),
        ("NOCount",            "number of N + O atoms"),
    ],

    # ── Polarity & Partial Charges ─────────────────────────────────────────
    "Polarity & Partial Charges": [
        ("MaxPartialCharge",   "largest positive Gasteiger partial charge"),
        ("MinPartialCharge",   "largest negative (most negative) Gasteiger partial charge"),
        ("MaxAbsPartialCharge","largest absolute Gasteiger partial charge"),
        ("MinAbsPartialCharge","smallest absolute Gasteiger partial charge"),
    ],

    # ── Electrotopological State ───────────────────────────────────────────
    "Electrotopological State (EState)": [
        ("MaxEStateIndex",     "maximum EState index across all atoms"),
        ("MinEStateIndex",     "minimum EState index across all atoms"),
        ("MaxAbsEStateIndex",  "maximum absolute EState index"),
        ("MinAbsEStateIndex",  "minimum absolute EState index"),
    ],

    # ── Topological Connectivity Indices (Chi) ─────────────────────────────
    "Connectivity Indices (Chi)": [
        ("Chi0",  "molecular connectivity index χ₀ (0th order)"),
        ("Chi1",  "molecular connectivity index χ₁ (1st order, path-2)"),
        ("Chi0n", "normalised χ₀ (carbon-normalised)"),
        ("Chi1n", "normalised χ₁"),
        ("Chi0v", "valence connectivity index χ₀ᵛ"),
        ("Chi1v", "valence connectivity index χ₁ᵛ"),
        ("Chi2n", "normalised χ₂"),
        ("Chi2v", "valence connectivity index χ₂ᵛ"),
        ("Chi3n", "normalised χ₃"),
        ("Chi3v", "valence connectivity index χ₃ᵛ"),
        ("Chi4n", "normalised χ₄"),
        ("Chi4v", "valence connectivity index χ₄ᵛ"),
    ],

    # ── Kier Shape Indices ─────────────────────────────────────────────────
    "Kier Shape Indices (Kappa)": [
        ("Kappa1",       "Kier shape index ¹κ (encodes cyclicity)"),
        ("Kappa2",       "Kier shape index ²κ (encodes spatial density)"),
        ("Kappa3",       "Kier shape index ³κ (encodes central volume)"),
        ("HallKierAlpha","Hall-Kier α (average atomic contribution / atom count)"),
        ("Phi",          "Kier molecular flexibility index φ"),
    ],

    # ── Other Topological Indices ──────────────────────────────────────────
    "Other Topological Indices": [
        ("BalabanJ","Balaban's J — average distance-sum connectivity index"),
        ("BertzCT", "Bertz complexity index (bonding + heteroatom complexity)"),
        ("Ipc",     "information content of the characteristic polynomial (total population)"),
        ("AvgIpc",  "average information content (Ipc / total population)"),
    ],

    # ── Ring Descriptors ───────────────────────────────────────────────────
    "Ring Descriptors": [
        ("RingCount",                 "total number of rings (SSSR)"),
        ("NumAromaticRings",          "number of aromatic rings"),
        ("NumSaturatedRings",         "number of saturated rings"),
        ("NumAliphaticRings",         "number of aliphatic rings (≥ 1 non-aromatic bond)"),
        ("NumAromaticCarbocycles",    "number of aromatic carbocycles"),
        ("NumAromaticHeterocycles",   "number of aromatic heterocycles"),
        ("NumSaturatedCarbocycles",   "number of saturated carbocycles"),
        ("NumSaturatedHeterocycles",  "number of saturated heterocycles"),
        ("NumAliphaticCarbocycles",   "number of aliphatic carbocycles"),
        ("NumAliphaticHeterocycles",  "number of aliphatic heterocycles"),
        ("NumHeterocycles",           "total number of heterocycles"),
    ],

    # ── Atom / Group Counts ────────────────────────────────────────────────
    "Atom & Group Counts": [
        ("NumRotatableBonds",        "number of rotatable bonds"),
        ("NumHeteroatoms",           "number of heteroatoms (non-C, non-H)"),
        ("NumAmideBonds",            "number of amide bonds"),
        ("NumBridgeheadAtoms",       "number of bridgehead atoms"),
        ("NumSpiroAtoms",            "number of spiro atoms"),
        ("NumAtomStereoCenters",     "total number of atomic stereocenters"),
        ("NumUnspecifiedAtomStereoCenters","number of unspecified atomic stereocenters"),
        ("NumRadicalElectrons",      "number of radical electrons"),
        ("FractionCSP3",             "fraction of sp³-hybridised carbon atoms"),
    ],

    # ── VSA : EState ───────────────────────────────────────────────────────
    "VSA EState (11)": [
        ("EState_VSA1",  "EState VSA 1  (-∞ < x < -0.39)"),
        ("EState_VSA2",  "EState VSA 2  (-0.39 ≤ x < 0.29)"),
        ("EState_VSA3",  "EState VSA 3  (0.29 ≤ x < 0.72)"),
        ("EState_VSA4",  "EState VSA 4  (0.72 ≤ x < 1.17)"),
        ("EState_VSA5",  "EState VSA 5  (1.17 ≤ x < 1.54)"),
        ("EState_VSA6",  "EState VSA 6  (1.54 ≤ x < 1.81)"),
        ("EState_VSA7",  "EState VSA 7  (1.81 ≤ x < 2.05)"),
        ("EState_VSA8",  "EState VSA 8  (2.05 ≤ x < 4.69)"),
        ("EState_VSA9",  "EState VSA 9  (4.69 ≤ x < 9.17)"),
        ("EState_VSA10", "EState VSA 10 (9.17 ≤ x < 15.00)"),
        ("EState_VSA11", "EState VSA 11 (15.00 ≤ x < ∞)"),
    ],

    # ── VSA : VSA EState ───────────────────────────────────────────────────
    "VSA EState (atom-type, 10)": [
        ("VSA_EState1",  "VSA EState 1  (-∞ < x < 4.78)"),
        ("VSA_EState2",  "VSA EState 2  (4.78 ≤ x < 5.00)"),
        ("VSA_EState3",  "VSA EState 3  (5.00 ≤ x < 5.41)"),
        ("VSA_EState4",  "VSA EState 4  (5.41 ≤ x < 5.74)"),
        ("VSA_EState5",  "VSA EState 5  (5.74 ≤ x < 6.00)"),
        ("VSA_EState6",  "VSA EState 6  (6.00 ≤ x < 6.07)"),
        ("VSA_EState7",  "VSA EState 7  (6.07 ≤ x < 6.45)"),
        ("VSA_EState8",  "VSA EState 8  (6.45 ≤ x < 7.00)"),
        ("VSA_EState9",  "VSA EState 9  (7.00 ≤ x < 11.00)"),
        ("VSA_EState10", "VSA EState 10 (11.00 ≤ x < ∞)"),
    ],

    # ── VSA : PEOE ─────────────────────────────────────────────────────────
    "VSA PEOE (partial-charge, 14)": [
        ("PEOE_VSA1",  "PEOE VSA 1  (-∞ < x < -0.30)"),
        ("PEOE_VSA2",  "PEOE VSA 2  (-0.30 ≤ x < -0.25)"),
        ("PEOE_VSA3",  "PEOE VSA 3  (-0.25 ≤ x < -0.20)"),
        ("PEOE_VSA4",  "PEOE VSA 4  (-0.20 ≤ x < -0.15)"),
        ("PEOE_VSA5",  "PEOE VSA 5  (-0.15 ≤ x < -0.10)"),
        ("PEOE_VSA6",  "PEOE VSA 6  (-0.10 ≤ x < -0.05)"),
        ("PEOE_VSA7",  "PEOE VSA 7  (-0.05 ≤ x < 0.00)"),
        ("PEOE_VSA8",  "PEOE VSA 8  (0.00 ≤ x < 0.05)"),
        ("PEOE_VSA9",  "PEOE VSA 9  (0.05 ≤ x < 0.10)"),
        ("PEOE_VSA10", "PEOE VSA 10 (0.10 ≤ x < 0.15)"),
        ("PEOE_VSA11", "PEOE VSA 11 (0.15 ≤ x < 0.20)"),
        ("PEOE_VSA12", "PEOE VSA 12 (0.20 ≤ x < 0.25)"),
        ("PEOE_VSA13", "PEOE VSA 13 (0.25 ≤ x < 0.30)"),
        ("PEOE_VSA14", "PEOE VSA 14 (0.30 ≤ x < ∞)"),
    ],

    # ── VSA : SLogP ────────────────────────────────────────────────────────
    "VSA SLogP (12)": [
        ("SlogP_VSA1",  "SLogP VSA 1  (-∞ < x < -0.40)"),
        ("SlogP_VSA2",  "SLogP VSA 2  (-0.40 ≤ x < -0.20)"),
        ("SlogP_VSA3",  "SLogP VSA 3  (-0.20 ≤ x < 0.00)"),
        ("SlogP_VSA4",  "SLogP VSA 4  (0.00 ≤ x < 0.10)"),
        ("SlogP_VSA5",  "SLogP VSA 5  (0.10 ≤ x < 0.15)"),
        ("SlogP_VSA6",  "SLogP VSA 6  (0.15 ≤ x < 0.20)"),
        ("SlogP_VSA7",  "SLogP VSA 7  (0.20 ≤ x < 0.25)"),
        ("SlogP_VSA8",  "SLogP VSA 8  (0.25 ≤ x < 0.30)"),
        ("SlogP_VSA9",  "SLogP VSA 9  (0.30 ≤ x < 0.40)"),
        ("SlogP_VSA10", "SLogP VSA 10 (0.40 ≤ x < 0.50)"),
        ("SlogP_VSA11", "SLogP VSA 11 (0.50 ≤ x < 0.60)"),
        ("SlogP_VSA12", "SLogP VSA 12 (0.60 ≤ x < ∞)"),
    ],

    # ── VSA : SMR ──────────────────────────────────────────────────────────
    "VSA SMR (10)": [
        ("SMR_VSA1",  "SMR VSA 1  (-∞ < x < 1.29)"),
        ("SMR_VSA2",  "SMR VSA 2  (1.29 ≤ x < 1.82)"),
        ("SMR_VSA3",  "SMR VSA 3  (1.82 ≤ x < 2.24)"),
        ("SMR_VSA4",  "SMR VSA 4  (2.24 ≤ x < 2.45)"),
        ("SMR_VSA5",  "SMR VSA 5  (2.45 ≤ x < 2.75)"),
        ("SMR_VSA6",  "SMR VSA 6  (2.75 ≤ x < 3.05)"),
        ("SMR_VSA7",  "SMR VSA 7  (3.05 ≤ x < 3.63)"),
        ("SMR_VSA8",  "SMR VSA 8  (3.63 ≤ x < 3.80)"),
        ("SMR_VSA9",  "SMR VSA 9  (3.80 ≤ x < 4.00)"),
        ("SMR_VSA10", "SMR VSA 10 (4.00 ≤ x < ∞)"),
    ],

    # ── BCUT ───────────────────────────────────────────────────────────────
    "BCUT (Burden-CAS-University-of-Texas, 8)": [
        ("BCUT2D_MWHI",   "BCUT high  — atomic mass weighted"),
        ("BCUT2D_MWLOW",  "BCUT low   — atomic mass weighted"),
        ("BCUT2D_CHGHI",  "BCUT high  — Gasteiger charge weighted"),
        ("BCUT2D_CHGLO",  "BCUT low   — Gasteiger charge weighted"),
        ("BCUT2D_LOGPHI", "BCUT high  — Wildman-Crippen LogP weighted"),
        ("BCUT2D_LOGPLOW","BCUT low   — Wildman-Crippen LogP weighted"),
        ("BCUT2D_MRHI",   "BCUT high  — Wildman-Crippen MR weighted"),
        ("BCUT2D_MRLOW",  "BCUT low   — Wildman-Crippen MR weighted"),
    ],

    # ── Surface Area ───────────────────────────────────────────────────────
    "Surface Area": [
        ("TPSA",        "topological polar surface area (N + O + S + P contribution)"),
        ("LabuteASA",   "Labute's approximate surface area"),
        ("SPS",         "spacial score — steric accessibility per heavy atom"),
    ],

    # ── Fingerprint Density ────────────────────────────────────────────────
    "Fingerprint Density": [
        ("FpDensityMorgan1", "Morgan fingerprint bit density (radius=1)"),
        ("FpDensityMorgan2", "Morgan fingerprint bit density (radius=2)"),
        ("FpDensityMorgan3", "Morgan fingerprint bit density (radius=3)"),
    ],

    # ── Drug-likeness ──────────────────────────────────────────────────────
    "Drug-likeness": [
        ("qed", "quantitative estimate of drug-likeness (Bickerton et al., 2012)"),
    ],

    # ── Fragment / Substructure Counts ─────────────────────────────────────
    "Fragment Counts (functional groups, ~80)": [
        ("fr_Al_COO",              "aliphatic carboxylic acids"),
        ("fr_Al_OH",               "aliphatic hydroxyl groups"),
        ("fr_Al_OH_noTert",        "aliphatic hydroxyl groups (excl. tert-OH)"),
        ("fr_ArN",                 "N functional groups attached to aromatics"),
        ("fr_Ar_COO",              "aromatic carboxylic acids"),
        ("fr_Ar_N",                "aromatic nitrogens"),
        ("fr_Ar_NH",               "aromatic amines"),
        ("fr_Ar_OH",               "aromatic hydroxyl groups"),
        ("fr_COO",                 "carboxylic acids"),
        ("fr_COO2",                "carboxylic acids (alternate)"),
        ("fr_C_O",                 "carbonyl O"),
        ("fr_C_O_noCOO",           "carbonyl O (excl. COOH)"),
        ("fr_C_S",                 "thiocarbonyl"),
        ("fr_HOCCN",               "C(OH)CCN-Ctert-alkyl or C(OH)CCNcyclic"),
        ("fr_Imine",               "imines"),
        ("fr_NH0",                 "tertiary amines"),
        ("fr_NH1",                 "secondary amines"),
        ("fr_NH2",                 "primary amines"),
        ("fr_N_O",                 "hydroxylamine groups"),
        ("fr_Ndealkylation1",      "XCCNR groups"),
        ("fr_Ndealkylation2",      "tert-alicyclic amines"),
        ("fr_Nhpyrrole",           "H-pyrrole nitrogens"),
        ("fr_SH",                  "thiol groups"),
        ("fr_aldehyde",            "aldehydes"),
        ("fr_alkyl_carbamate",     "alkyl carbamates"),
        ("fr_alkyl_halide",        "alkyl halides"),
        ("fr_allylic_oxid",        "allylic oxidation sites"),
        ("fr_amide",               "amides"),
        ("fr_amidine",             "amidine groups"),
        ("fr_aniline",             "anilines"),
        ("fr_aryl_methyl",         "aryl methyl sites"),
        ("fr_azide",               "azide groups"),
        ("fr_azo",                 "azo groups"),
        ("fr_barbitur",            "barbiturate groups"),
        ("fr_benzene",             "benzene rings"),
        ("fr_benzodiazepine",      "benzodiazepines"),
        ("fr_bicyclic",            "bicyclic"),
        ("fr_diazo",               "diazo groups"),
        ("fr_dihydropyridine",     "dihydropyridines"),
        ("fr_epoxide",             "epoxide rings"),
        ("fr_ester",               "esters"),
        ("fr_ether",               "ether oxygens (incl. phenoxy)"),
        ("fr_furan",               "furan rings"),
        ("fr_guanido",             "guanidine groups"),
        ("fr_halogen",             "halogens"),
        ("fr_hdrzine",             "hydrazine groups"),
        ("fr_hdrzone",             "hydrazone groups"),
        ("fr_imidazole",           "imidazole rings"),
        ("fr_imide",               "imide groups"),
        ("fr_isocyan",             "isocyanates"),
        ("fr_isothiocyan",         "isothiocyanates"),
        ("fr_ketone",              "ketones"),
        ("fr_ketone_Topliss",      "ketones (Topliss subset)"),
        ("fr_lactam",              "β-lactams"),
        ("fr_lactone",             "lactones (cyclic esters)"),
        ("fr_methoxy",             "methoxy groups"),
        ("fr_morpholine",          "morpholine rings"),
        ("fr_nitrile",             "nitriles"),
        ("fr_nitro",               "nitro groups"),
        ("fr_nitro_arom",          "nitro benzene ring substituents"),
        ("fr_nitro_arom_nonortho", "non-ortho nitro benzene substituents"),
        ("fr_nitroso",             "nitroso groups (excl. NO₂)"),
        ("fr_oxazole",             "oxazole rings"),
        ("fr_oxime",               "oxime groups"),
        ("fr_para_hydroxylation",  "para-hydroxylation sites"),
        ("fr_phenol",              "phenols"),
        ("fr_phenol_noOrthoHbond", "phenolic OH (excl. ortho intramol. H-bond)"),
        ("fr_phos_acid",           "phosphoric acid groups"),
        ("fr_phos_ester",          "phosphoric ester groups"),
        ("fr_piperdine",           "piperidine rings"),
        ("fr_piperzine",           "piperazine rings"),
        ("fr_priamide",            "primary amides"),
        ("fr_prisulfonamd",        "primary sulfonamides"),
        ("fr_pyridine",            "pyridine rings"),
        ("fr_quatN",               "quaternary nitrogens"),
        ("fr_sulfide",             "thioethers"),
        ("fr_sulfonamd",           "sulfonamides"),
        ("fr_sulfone",             "sulfone groups"),
        ("fr_term_acetylene",      "terminal acetylenes"),
        ("fr_tetrazole",           "tetrazole rings"),
        ("fr_thiazole",            "thiazole rings"),
        ("fr_thiocyan",            "thiocyanates"),
        ("fr_thiophene",           "thiophene rings"),
        ("fr_unbrch_alkane",       "unbranched alkanes (≥ 4 C)"),
        ("fr_urea",                "urea groups"),
    ],
}
