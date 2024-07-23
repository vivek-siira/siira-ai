use Chinook;
select a.ArtistId, a.Name , count(pt.TrackId) from 
PlaylistTrack pt inner join Track t on pt.TrackId = t.TrackId
inner join Album alb on alb.AlbumId = t.AlbumId
inner join Artist a on a.ArtistId = alb.ArtistId
group by 1,2 order by count(pt.TrackId) desc limit 1;