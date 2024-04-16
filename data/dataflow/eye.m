Dataflow {
            TemporalMap(CTileSz,CTileSz) C;
			TemporalMap(KTileSz,KTileSz) K;
			SpatialMap(Sz(R),Sz(R)) Y';
			TemporalMap(Sz(S),Sz(S)) X';
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			Cluster(ClusterSz, P);
			TemporalMap(1,1) C;
			SpatialMap(1,1) Y';
			SpatialMap(1,1) R;
			TemporalMap(Sz(R),Sz(R)) X;
			TemporalMap(Sz(S),Sz(S)) S;
		}
