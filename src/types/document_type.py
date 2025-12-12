from enum import Enum

class DocumentType(Enum):
     COMMISSION_AGENDA = "commission_agenda"
     DISCOVERY_PACKAGE = "discovery_package"
     PRESS_RELEASE = "press_release"
     CORRESPONDENCE = "correspondence"
     REPORTS = "reports"
     UNKNOWN = "unknown"
