import { Info } from "lucide-react";


function InfoBanner({ children }) {

  return (

    <div className="info-banner">

      <Info size={18} />

      <div>
        {children}
      </div>

    </div>

  );

}


export default InfoBanner;
