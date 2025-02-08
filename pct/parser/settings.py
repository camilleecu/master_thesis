from pct.parser.section import Section

class Settings:
    def __init__(self, filename):
        self.filename = filename
        self.sections = {}
        self.initializeDefault()

    def addSection(self, section):
        self.sections[section.name] = section

    def initializeDefault(self):
        """Initialize all sections with their default values."""
        # ===============
        # General section
        section = Section("General")
        section.addAttribute("Verbose"   , 1)
        section.addAttribute("RandomSeed", None, dtype=int)
        self.addSection(section)

        # ============
        # Data section
        section = Section("Data")
        section.addAttribute("File"     , "") # Default: self.filename.strip(".s") + ".csv" ?
        section.addAttribute("TestSet"  , "")
        section.addAttribute("Hierarchy", "") # Default: Path(str(os.path.splitext(File)[0]) + ".hierarchy.txt")?
        self.addSection(section)

        # ==================
        # Attributes section
        section = Section("Attributes")
        section.addAttribute("Target"     , None, dtype=list)
        # section.addAttribute("Descriptive", [-1]) # TODO put this in out file (but not as setting)
        self.addSection(section)

        # ==============
        # Output section
        section = Section("Output")
        section.addAttribute("WritePredictionsTrain", False)
        section.addAttribute("WritePredictionsTest" , False)
        self.addSection(section)

        # =============
        # Model section
        section = Section("Model")
        section.addAttribute("MinimalWeight", 5.0)
        self.addSection(section)

        # ============
        # Tree section
        section = Section("Tree")
        # section.addAttribute("Heuristic", "VarianceReduction")
        section.addAttribute("FTest", 0.01)
        section.addAttribute("PBCT" , False)
        self.addSection(section)

        # ====================
        # Hierarchical section
        section = Section("Hierarchical")
        section.addAttribute("Type"      , None, dtype=str)
        section.addAttribute("WType"     , "ExpAvgParentWeight")
        section.addAttribute("WParam"    , 0.75)
        # section.addAttribute("HSeparator", "/")
        section.addAttribute("ClassificationThreshold", list(range(0,102,2)))
        self.addSection(section)

        # ==============
        # Forest section
        section = Section("Forest")
        section.addAttribute("NumberOfTrees"           , None, dtype=int)
        section.addAttribute("BagSize"                 , -1, dtype=int)
        section.addAttribute("NumberOfFeatures"        , -1, dtype=int)
        section.addAttribute("NumberOfVerticalFeatures", -1, dtype=int)
        section.addAttribute("FeatureRanking"          , False)
        section.addAttribute("OOBEstimate"             , False, dtype=int)
        self.addSection(section)

    def __str__(self):
        string = ""
        for section in self.sections.values():
            string += str(section) + "\n"
        return string[:-2] # Remove last newline
